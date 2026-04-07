import httpx
from typing import Optional
from app.core.config import settings
from app.schemas.ai import TransactionRequest, FraudVerdict

class AIClient:
    def __init__(self, base_url: str = settings.AI_SERVICE_URL):
        self.base_url = base_url

    async def score_transaction(self, request: TransactionRequest) -> Optional[FraudVerdict]:
        async with httpx.AsyncClient(timeout=10.0) as client:
            try:
                response = await client.post(
                    f"{self.base_url}/score",
                    json=request.model_dump()
                )
                response.raise_for_status()
                return FraudVerdict(**response.json())
            except httpx.HTTPError as e:
                print(f"Error calling AI Service: {e}")
                return None

# Global instance
ai_client = AIClient()
