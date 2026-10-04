import { useMemo, useState } from 'react'

const API_BASE = import.meta.env.VITE_API_BASE_URL || '/api'

const EXAMPLE_QUESTIONS = [
  '3号泵启动以后温度越来越高，现在已经出现振动告警，帮我分析一下。',
  '检查 device-003 当前状态，并判断是否存在过热风险。',
  '请分析最近30分钟设备异常日志，并给出处置建议。',
]

function formatJson(value) {
  if (typeof value === 'string') return value
  return JSON.stringify(value, null, 2)
}

function eventLabel(type) {
  const labels = {
    agent_start: 'Agent 开始',
    agent_step: 'Agent 思考步骤',
    tool_call: '调用工具',
    tool_result: '工具结果',
    tool_error: '工具异常',
    waiting_for_approval: '等待人工审批',
    human_rejected: '人工拒绝',
    approved_tool_result: '审批后工具结果',
    memory_saved: 'Memory 已保存',
    memory_save_error: 'Memory 保存异常',
    diagnosis: '生成诊断结果',
    agent_finish: 'Agent 完成',
    tool_not_found: '工具不存在',
    resume_tool_not_found: '恢复工具不存在',
    approved_tool_error: '审批后工具执行失败',
  }
  return labels[type] || type
}

function statusText(status) {
  const map = {
    idle: '空闲',
    running: '运行中',
    waiting_for_approval: '等待审批',
    completed: '已完成',
    failed: '执行失败',
  }
  return map[status] || status
}

function RiskBadge({ risk }) {
  if (!risk) return null
  const normalized = String(risk).toLowerCase()
  return <span className={`risk-badge risk-${normalized}`}>{String(risk).toUpperCase()}</span>
}

function DiagnosisCard({ diagnosis }) {
  if (!diagnosis) return null

  return (
    <section className="card diagnosis-card">
      <div className="section-title-row">
        <h2>诊断结果</h2>
        <RiskBadge risk={diagnosis.risk_level} />
      </div>

      <div className="diagnosis-summary">{diagnosis.summary || '暂无总结'}</div>

      <div className="diagnosis-grid">
        <div>
          <div className="field-label">故障类型</div>
          <div className="field-value">{diagnosis.fault_type || '未知'}</div>
        </div>
        <div>
          <div className="field-label">置信度</div>
          <div className="field-value">
            {typeof diagnosis.confidence === 'number'
              ? `${Math.round(diagnosis.confidence * 100)}%`
              : '-'}
          </div>
        </div>
        <div>
          <div className="field-label">人工确认</div>
          <div className="field-value">
            {diagnosis.requires_human_confirmation ? '需要' : '不需要'}
          </div>
        </div>
      </div>

      <div className="two-col">
        <div>
          <div className="field-label">诊断依据</div>
          <ul className="compact-list">
            {(diagnosis.evidence || []).map((item, index) => (
              <li key={index}>{item}</li>
            ))}
          </ul>
        </div>
        <div>
          <div className="field-label">建议措施</div>
          <ul className="compact-list">
            {(diagnosis.recommended_actions || []).map((item, index) => (
              <li key={index}>{item}</li>
            ))}
          </ul>
        </div>
      </div>

      {Array.isArray(diagnosis.sources) && diagnosis.sources.length > 0 && (
        <div>
          <div className="field-label">知识来源</div>
          <div className="tag-list">
            {diagnosis.sources.map((source) => (
              <span className="tag" key={source}>{source}</span>
            ))}
          </div>
        </div>
      )}
    </section>
  )
}

function TracePanel({ trace }) {
  return (
    <section className="card">
      <div className="section-title-row">
        <h2>Agent Trace</h2>
        <span className="muted">{trace.length} 个事件</span>
      </div>
      {trace.length === 0 ? (
        <div className="empty">暂无执行轨迹</div>
      ) : (
        <div className="trace-list">
          {trace.map((item, index) => (
            <details className="trace-item" key={`${item.event || 'event'}-${index}`} open={index === trace.length - 1}>
              <summary>
                <span className="trace-index">{index + 1}</span>
                <span>{eventLabel(item.event)}</span>
                <span className="trace-type">{item.event}</span>
              </summary>
              <pre>{formatJson(item.data ?? {})}</pre>
            </details>
          ))}
        </div>
      )}
    </section>
  )
}

function ApprovalCard({ pendingTool, runId, loading, onApprove, onReject }) {
  const [reason, setReason] = useState('')

  if (!pendingTool) return null

  return (
    <section className="card approval-card">
      <div className="approval-header">
        <div>
          <div className="approval-kicker">需要人工确认</div>
          <h2>{pendingTool.tool_name}</h2>
        </div>
        <RiskBadge risk={pendingTool.risk_level} />
      </div>

      <p className="approval-text">
        Agent 请求执行一个需要人工批准的工具。批准后，后端会恢复当前 Agent Run，
        执行这个工具，并继续后面的 Agent Loop。
      </p>

      <div className="approval-meta">
        <div>
          <div className="field-label">Run ID</div>
          <code>{runId}</code>
        </div>
        <div>
          <div className="field-label">参数</div>
          <pre>{formatJson(pendingTool.arguments || {})}</pre>
        </div>
      </div>

      <div className="reject-box">
        <label htmlFor="reject-reason">拒绝原因（可选）</label>
        <textarea
          id="reject-reason"
          value={reason}
          onChange={(e) => setReason(e.target.value)}
          placeholder="例如：当前设备正在生产任务中，不允许重启。"
          rows={3}
          disabled={loading}
        />
      </div>

      <div className="approval-actions">
        <button
          className="button danger-outline"
          disabled={loading}
          onClick={() => onReject(reason || '用户拒绝执行')}
        >
          {loading ? '处理中...' : '拒绝执行'}
        </button>
        <button
          className="button primary"
          disabled={loading}
          onClick={onApprove}
        >
          {loading ? '执行中...' : '批准并继续'}
        </button>
      </div>
    </section>
  )
}

export default function App() {
  const [message, setMessage] = useState('')
  const [deviceId, setDeviceId] = useState('device-003')
  const [status, setStatus] = useState('idle')
  const [runId, setRunId] = useState('')
  const [pendingTool, setPendingTool] = useState(null)
  const [diagnosis, setDiagnosis] = useState(null)
  const [trace, setTrace] = useState([])
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)

  const statusClass = useMemo(() => `status-chip status-${status}`, [status])

  function mergeResponse(data) {
    setRunId(data.session_id || '')
    setStatus(data.status || 'idle')
    setPendingTool(data.pending_tool || null)
    setDiagnosis(data.diagnosis || null)
    setTrace(data.trace || [])
  }

  async function request(path, options = {}) {
    const response = await fetch(`${API_BASE}${path}`, {
      headers: {
        'Content-Type': 'application/json',
        ...(options.headers || {}),
      },
      ...options,
    })

    const raw = await response.text()
    let data = null
    try {
      data = raw ? JSON.parse(raw) : {}
    } catch {
      data = { detail: raw }
    }

    if (!response.ok) {
      const detail = data?.detail || data?.error || `HTTP ${response.status}`
      throw new Error(detail)
    }

    return data
  }

  async function handleRun() {
    if (!message.trim() || loading) return

    setError('')
    setDiagnosis(null)
    setPendingTool(null)
    setTrace([])
    setRunId('')
    setStatus('running')
    setLoading(true)

    try {
      const data = await request('/agent/run', {
        method: 'POST',
        body: JSON.stringify({
          message: message.trim(),
          device_id: deviceId.trim() || null,
        }),
      })

      mergeResponse(data)
    } catch (err) {
      setStatus('failed')
      setError(err.message || '请求失败')
    } finally {
      setLoading(false)
    }
  }

  async function resumeAgent(approved, reason) {
    if (!runId || loading) return

    setError('')
    setLoading(true)

    try {
      const data = await request(`/agent/${encodeURIComponent(runId)}/resume`, {
        method: 'POST',
        body: JSON.stringify({ approved, reason }),
      })
      mergeResponse(data)
    } catch (err) {
      setStatus('failed')
      setError(err.message || '审批请求失败')
    } finally {
      setLoading(false)
    }
  }

  function useExample(text) {
    setMessage(text)
  }

  function reset() {
    setMessage('')
    setStatus('idle')
    setRunId('')
    setPendingTool(null)
    setDiagnosis(null)
    setTrace([])
    setError('')
  }

  return (
    <div className="app-shell">
      <header className="topbar">
        <div>
          <div className="eyebrow">AI OPS / REACT AGENT</div>
          <h1>智能运维故障诊断 Agent</h1>
          <p className="subtitle">RAG + 多工具协作 + Memory + Human-in-the-loop + DiagnosisResult</p>
        </div>
        <div className={statusClass}>{statusText(status)}</div>
      </header>

      <main className="content">
        <section className="card input-card">
          <div className="section-title-row">
            <h2>发起诊断</h2>
            <span className="muted">API: {API_BASE}</span>
          </div>

          <div className="form-grid">
            <label>
              <span>设备 ID</span>
              <input
                value={deviceId}
                onChange={(e) => setDeviceId(e.target.value)}
                placeholder="device-003"
                disabled={loading}
              />
            </label>
            <label className="message-field">
              <span>故障描述</span>
              <textarea
                value={message}
                onChange={(e) => setMessage(e.target.value)}
                placeholder="例如：3号泵启动以后温度越来越高，现在已经出现振动告警，帮我分析一下。"
                rows={4}
                disabled={loading}
              />
            </label>
          </div>

          <div className="examples">
            <span className="muted">示例：</span>
            {EXAMPLE_QUESTIONS.map((item) => (
              <button
                key={item}
                className="example-chip"
                onClick={() => useExample(item)}
                disabled={loading}
              >
                {item}
              </button>
            ))}
          </div>

          <div className="toolbar">
            <span className="muted">
              {runId ? `Run ID: ${runId}` : '尚未创建 Agent Run'}
            </span>
            <div className="toolbar-actions">
              <button className="button ghost" onClick={reset} disabled={loading}>清空</button>
              <button className="button primary" onClick={handleRun} disabled={loading || !message.trim()}>
                {loading ? 'Agent 执行中...' : '开始诊断'}
              </button>
            </div>
          </div>
        </section>

        {error && (
          <section className="card error-card">
            <strong>请求异常</strong>
            <div>{error}</div>
          </section>
        )}

        {pendingTool && (
          <ApprovalCard
            pendingTool={pendingTool}
            runId={runId}
            loading={loading}
            onApprove={() => resumeAgent(true)}
            onReject={(reason) => resumeAgent(false, reason)}
          />
        )}

        {diagnosis && <DiagnosisCard diagnosis={diagnosis} />}

        <TracePanel trace={trace} />
      </main>
    </div>
  )
}
