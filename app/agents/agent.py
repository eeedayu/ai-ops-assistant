import json
import logging
from typing import Any

from app.agents.registry import ToolRegistry
from app.agents.memory import LongTermMemory
from app.agents.state_store import AgentStateStore
from app.agents.state import (
    AgentState,
    ToolCallRecord,
    AgentEvent,
)
from app.core.config import get_settings
from app.core.exceptions import AppException
from app.repositories.session_repository import (
    SessionRepository,
)
from app.services.llm_service import LLMService


logger = logging.getLogger(__name__)


AGENT_SYSTEM_PROMPT = """
                            你是一名 AI 运维故障诊断助手。

                            你的任务：
                            1. 理解用户描述的故障现象
                            2. 判断需要哪些信息
                            3. 使用工具获取证据
                            4. 使用知识库查询相关运维知识
                            5. 综合多个证据进行故障诊断
                            6. 给出风险等级和处理建议
                            7. 涉及高风险操作时必须请求人工确认

                            规则：
                            - 不要凭空编造设备状态
                            - 不要把知识库内容当成实时状态
                            - 不要把推测当成事实
                            - 诊断结论必须给出证据
                            - 信息不足时继续调用工具
                            - 最终输出结构化 DiagnosisResult
                            """


class AgentService:

    def __init__(
        self,
        session_repository: SessionRepository,
        llm_service: LLMService,
        tool_registry: ToolRegistry,
        memory: LongTermMemory,
        state_store: AgentStateStore,
    ):
        self.session_repository = (
            session_repository
        )

        self.llm_service = llm_service

        self.tool_registry = tool_registry

        self.memory = memory

        self.state_store = state_store

        self.settings = get_settings()

    def _build_messages(
        self,
        session_id: str,
        user_message: str,
    ) -> list[dict[str, Any]]:

        history = (
            self.session_repository
            .get_messages(session_id)
        )

        if not history:

            history = [
                {
                    "role": "system",
                    "content": AGENT_SYSTEM_PROMPT,
                }
            ]

        history.append(
            {
                "role": "user",
                "content": user_message,
            }
        )

        return history

    @staticmethod
    def _tool_signature(
        tool_name: str,
        arguments: dict[str, Any],
    ) -> str:

        args_json = json.dumps(arguments, sort_keys=True, ensure_ascii=False)
        return f"{tool_name}:{args_json}"

    async def run(
        self,
        session_id: str,
        user_message: str,
    ) -> AgentState:

        state = AgentState(
            session_id=session_id
        )

        state.trace.append(
            AgentEvent(
                type="agent_start",
                step=state.step_count,
                message="Agent开始运行",
            )
        )

        self.session_repository.get_or_create(
            session_id
        )

        messages = self._build_messages(
            session_id=session_id,
            user_message=user_message,
        )

        self.session_repository.add_message(
            session_id=session_id,
            role="user",
            content=user_message,
        )

        memories = await self.memory.search(
            query=user_message,
            top_k=5,
        )

        if memories:

            memory_text = "\n\n".join(
                (
                    f"[历史诊断 {index}]\n"
                    f"{item.content}"
                )
                for index, item in enumerate(
                    memories,
                    start=1,
                )
            )

            messages.append({
                "role": "system",
                "content": (
                    "以下是与当前问题相关的历史诊断记录，"
                    "只能作为辅助证据，不能直接当成当前设备状态：\n\n"
                    + memory_text
                ),
            })

        state.messages = messages

        await self.state_store.save(state)

        logger.info(
            "Agent started session=%s",
            session_id,
        )

        return await self._continue(
            state
        )
    
    async def _continue(
        self,
        state: AgentState,
    ):
        while (
            state.step_count
            < self.settings.agent_max_steps
        ):

            state.step_count += 1

            logger.info(
                "Agent step=%s session=%s",
                state.step_count,
                state.session_id,
            )

            try:

                assistant_message = (
                    await self.llm_service
                    .chat_with_tools(
                        messages=state.messages,
                        tools=self.tool_registry.schemas(),
                    )
                )

            except Exception as exc:

                state.status = "failed"

                state.error = str(exc)

                logger.exception(
                    "Agent LLM failed"
                )

                return state

            state.messages.append(
                assistant_message
            )

            tool_calls = (
                assistant_message
                .get("tool_calls", [])
            )

            # 没有工具调用
            if not tool_calls:

                try:

                    diagnosis = (
                        await self
                        ._generate_final_diagnosis(
                            state.messages
                        )
                    )
                    state.diagnosis = diagnosis
                    state.final_answer = (
                        assistant_message.get(
                            "content",
                            "",
                        )
                    )

                    try:
                        await self.memory.save_diagnosis(
                            diagnosis=diagnosis,
                        )

                        state.trace.append(
                            AgentEvent(
                                type="memory_saved",
                                step=state.step_count,
                                message="完成当前故障诊断记忆存档",
                                data={
                                    "fault_type":
                                        diagnosis.fault_type
                                },
                            )
                        )
                    except Exception as e:
                        state.trace.append(
                            AgentEvent(
                                type="memory_save_error",
                                step=state.step_count,
                                message="故障诊断记忆存档失败",
                                data={
                                    "error": str(e)
                                },
                            )
                        )

                    state.status = "completed"
                    state.trace.append(
                        AgentEvent(
                            type="diagnosis_complete",
                            step=state.step_count,
                            message="完成故障诊断",
                        )
                    )

                    await self.state_store.save(
                                state
                            )

                    return state

                except Exception as exc:

                    state.status = "failed"

                    state.error = str(exc)

                    return state

            # 有工具调用
            for tool_call in tool_calls:
                
                tool_call_id = (
                    tool_call["id"]
                )

                function = (
                    tool_call["function"]
                )

                tool_name = (
                    function["name"]
                )

                raw_arguments = (
                    function.get(
                        "arguments",
                        "{}",
                    )
                )

                if self.tool_registry.get(tool_name).requires_approval:

                    state.status = (
                        "waiting_for_approval"
                    )

                    state.pending_approval_tool = {
                        "tool_call_id": tool_call_id,
                        "tool_name": tool_name,
                        "arguments": raw_arguments,
                        "risk_level": self.tool_registry.get(tool_name).risk_level,
                    }

                    state.trace.append(
                        AgentEvent(
                            type="waiting_for_approval",
                            step=state.step_count,
                            message="等待调用工具中",
                            data=state.pending_approval_tool,
                        )
                    )

                    await self.state_store.save(
                        state
                    )

                    return state

                try:

                    arguments = json.loads(
                        raw_arguments
                    )

                except json.JSONDecodeError:

                    tool_result = {
                        "ok": False,
                        "error": (
                            "工具参数不是合法JSON"
                        ),
                    }

                    state.messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": (
                                tool_call_id
                            ),
                            "content": json.dumps(
                                tool_result,
                                ensure_ascii=False,
                            ),
                        }
                    )

                    continue

                signature = (
                    self._tool_signature(
                        tool_name,
                        arguments,
                    )
                )

                count = (
                    state
                    .repeated_tool_calls
                    .get(signature, 0)
                )

                if count >= 2:

                    tool_result = {
                        "ok": False,
                        "error": (
                            "检测到重复工具调用，"
                            "为避免Agent循环，"
                            "禁止继续执行相同操作。"
                        ),
                    }

                    state.messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": (
                                tool_call_id
                            ),
                            "content": json.dumps(
                                tool_result,
                                ensure_ascii=False,
                            ),
                        }
                    )

                    state.trace.append(
                        AgentEvent(
                            type="tool_result",
                            step=state.step_count,
                            message="检测到重复工具调用，禁止继续执行相同操作",
                            data={
                                "tool_info": ToolCallRecord(
                                                tool_call_id=tool_call_id,
                                                tool_name=tool_name,
                                                arguments=arguments,
                                                success=False,
                                                result=tool_result,                                    
                                            )
                            },
                        )
                    )

                    continue

                state.repeated_tool_calls[
                    signature
                ] = count + 1

                logger.info(
                    "Agent tool call "
                    "tool=%s args=%s",
                    tool_name,
                    arguments,
                )

                state = await self._execute_tool(
                    state,
                    tool_name=tool_name,
                    tool_call_id=tool_call_id,
                    arguments=arguments,
                )

                if state.status == "failed":
                    if state.step_count == self.settings.agent_max_steps:
                        await self.state_store.save(
                            state
                        )
                        return state
                    else:
                        state.status == "running"

        state.status = "max_steps"

        state.error = (
            "Agent达到最大执行步数"
        )

        await self.state_store.save(
            state
        )

        logger.warning(
            "Agent reached max steps "
            "session=%s",
            state.session_id,
        )

        return state

    async def _generate_final_diagnosis(
        self,
        messages: list[dict[str, Any]],
    ) :

        generate_diagnosis = getattr(
            self.llm_service,
            "generate_diagnosis",
            None,
        )

        if generate_diagnosis is None:
            return None

        diagnosis = (
            await generate_diagnosis(
                messages
            )
        )

        return diagnosis

    async def _execute_tool(
        self,
        state,
        tool_name,
        tool_call_id,
        arguments,
    ):
        state.trace.append(
            AgentEvent(
                type="tool_call",
                step=state.step_count,
                message=f"调用工具: {tool_name}",
                data={
                    "tool": tool_name,
                    "arguments": arguments
                },
            )
        )

        success = True
        tool_result = {}
        message=""

        try:

            result = (
                await self.tool_registry
                .execute(
                    name=tool_name,
                    arguments=arguments,
                )
            )
            message="调用工具成功"
            tool_result = {
                "ok": True,
                "data": result,
            }

        except Exception as e:

            success = False
            message="调用工具失败"
            tool_result = {
                "ok": False,
                "error": str(e),
            }
            state.status = "failed"
        
        state.trace.append(
            AgentEvent(
                type="tool_result",
                step=state.step_count,
                message=message,
                data={
                    "tool_info": ToolCallRecord(
                                    tool_call_id=tool_call_id,
                                    tool_name=tool_name,
                                    arguments=arguments,
                                    success=success,
                                    result=tool_result,                                    
                                )
                },
            )
        )

        state.messages.append(
            {
                "role": "tool",
                "tool_call_id": tool_call_id,
                "content": json.dumps(
                    tool_result,
                    ensure_ascii=False,
                ),
            }
        )

        return state

    async def approve(
        self,
        session_id: str,
    ):
        state = await self.state_store.get(
            session_id
        )

        if state is None:
            raise ValueError(
                "Agent Run 不存在"
            )

        if state.status != "waiting_for_approval":
            raise ValueError(
                "当前 Agent 不在等待审批状态"
            )

        return await self._resume_after_approval(
            state
        )
    
    async def _resume_after_approval(
        self,
        state: AgentState,
    ):

        pending_approval_tool = state.pending_approval_tool

        if pending_approval_tool is None:
            raise ValueError(
                "没有待审批的工具调用"
            )
        
        state = await self._execute_tool(
            state,
            tool_name=pending_approval_tool["tool_name"],
            tool_call_id=pending_approval_tool["tool_call_id"],
            arguments=pending_approval_tool["arguments"],
        )   

        if state.status == "failed":
            if state.step_count == self.settings.agent_max_steps:
                await self.state_store.save(
                    state
                )
                return state
            else:
                state.status == "running"
        

        state.pending_approval_tool = None

        await self.state_store.save(
            state
        )

        return await self._continue(
            state
        )

    async def reject(
            self,
            session_id: str,
        ):
            state = await self.state_store.get(
                session_id
            )
    
            if state is None:
                raise ValueError(
                    "Agent Run 不存在"
                )
    
            if state.status != "waiting_for_approval":
                raise ValueError(
                    "当前 Agent 不在等待审批状态"
                )

            data = {
                "tool_name": state.pending_approval_tool["tool_name"],
                "tool_call_id": state.pending_approval_tool["tool_call_id"],
                "arguments": state.pending_approval_tool["arguments"],
            }
            tool_result = {
                "ok": False,
                "message": "用户拒绝调用工具",
                "data": data,
            }

            state.trace.append(
                AgentEvent(
                    type="human_rejected",
                    step=state.step_count,
                    message="用户拒绝调用工具",
                    data=data,
                )
            )

            state.messages.append(
                {
                    "role": "tool",
                    "tool_call_id": state.pending_approval_tool["tool_call_id"],
                    "content": json.dumps(
                        tool_result,
                        ensure_ascii=False,
                    ),
                }
            )

            state.pending_approval_tool = None
            state.status = "running"
            
            await self.state_store.save(
                state
            )

            return await self._continue(
                state
            )