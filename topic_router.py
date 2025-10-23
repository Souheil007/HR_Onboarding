from transformers import pipeline

class TopicRouter:
    def __init__(self):
        # Small, fast zero-shot model for fallback
        self.classifier = pipeline(
            "zero-shot-classification",
            model="valhalla/distilbart-mnli-12-1",
            device=-1 
        )

        # Your topics
        self.labels = ["market_info", "contacts", "procedures"]

        # Simple keyword dictionary for fast routing
        self.keywords = {
            "market_info": [
                "stock", "market", "price", "trend", "analysis", "financial", "investment",
                "report", "growth", "revenue", "sales", "demand", "competition", "forecast",
                "valuation", "economy", "shares", "indices", "profit", "loss"
            ],
            "contacts": [
                "email", "contact", "phone", "reach", "person", "manager", "HR", "team",
                "representative", "support", "agent", "department", "directory", "extension",
                "connection", "colleague", "address", "network", "liaison"
            ],
            "procedures": [
                "policy", "process", "procedure", "guideline", "form", "manual", "instruction",
                "step", "compliance", "regulation", "protocol", "workflow", "approval",
                "submission", "documentation", "requirements", "checklist", "rule", "standard",
                "operation"
            ]
        }

    def detect_topic(self, query: str) -> str:
        query_lower = query.lower()

        # --- Step 1: Keyword-based routing ---
        for topic, kws in self.keywords.items():
            if any(kw in query_lower for kw in kws):
                return topic

        # --- Step 2: Fallback to zero-shot classification ---
        result = self.classifier(query, self.labels)
        topic = result["labels"][0]  # highest confidence
        return topic
