"""Offline C30 checkpoint demo. Run: python -m looplab.runner smoke --out smoke-run."""
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage


class ScriptedCatalogModel(FakeMessagesListChatModel):
    """Two predetermined messages per turn; the real graph dispatches the real fixture tool."""
    model_name: str = "deepseek:offline"

    def bind_tools(self, tools, **kwargs):
        return self


def catalog_model():
    return ScriptedCatalogModel(responses=[
        AIMessage(content="", tool_calls=[{"name": "refresh_local_catalog", "args": {}, "id": "refresh"}]),
        AIMessage(content="Catalog refresh attempted."),
    ])
