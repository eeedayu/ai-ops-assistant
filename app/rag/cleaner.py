import re


class TextCleaner:

    def clean(
        self,
        text: str,
    ) -> str:

        text = text.replace(
            "\r\n",
            "\n",
        )

        text = text.replace(
            "\r",
            "\n",
        )

        # 去掉连续空格
        text = re.sub(
            r"[ \t]+",
            " ",
            text,
        )

        # 去掉过多空行
        text = re.sub(
            r"\n{3,}",
            "\n\n",
            text,
        )

        return text.strip()