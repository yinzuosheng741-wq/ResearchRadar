"""Request-local workflow construction after the skill and permission checks."""


class LazyResearchWorkflow:
    def __init__(self, factory):
        self._factory = factory
        self._workflow = None

    def _get(self):
        if self._workflow is None:
            self._workflow = self._factory()
        return self._workflow

    @property
    def last_trace(self):
        # Diagnostics must never initialize an unused workflow.
        return getattr(self._workflow, "last_trace", None)

    @property
    def retriever(self):
        return getattr(self._workflow, "retriever", None)

    def answer(self, question, *, paper_ids=None, supplemental_queries=None):
        return self._get().answer(question, paper_ids=paper_ids, supplemental_queries=supplemental_queries)

    def plan(self, topic, *, paper_ids=None, supplemental_queries=None):
        return self._get().plan(topic, paper_ids=paper_ids, supplemental_queries=supplemental_queries)
