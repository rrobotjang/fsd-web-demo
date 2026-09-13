from fastapi import APIRouter
from pydantic import BaseModel
from typing import Optional
import random

router = APIRouter()

class PaymentRequest(BaseModel):
    scenario: str
    amount: Optional[float] = None

SCENARIO_RATES = {
    "toll": {"min": 2.50, "max": 8.00, "name": "Toll Gate"},
    "parking": {"min": 1.00, "max": 5.00, "name": "Parking"},
    "fuel": {"min": 30.00, "max": 80.00, "name": "Fuel Station"}
}

payment_history = []

@router.post("/payment")
async def process_payment(request: PaymentRequest):
    if request.scenario not in SCENARIO_RATES:
        return {"success": False, "message": f"Unknown scenario: {request.scenario}"}
    
    rate = SCENARIO_RATES[request.scenario]
    amount = request.amount or random.uniform(rate["min"], rate["max"])
    amount = round(amount, 2)
    
    payment_record = {
        "id": len(payment_history) + 1,
        "scenario": request.scenario,
        "scenario_name": rate["name"],
        "amount": amount,
        "status": "completed",
        "message": f"Payment of ${amount:.2f} processed for {rate['name']}"
    }
    payment_history.append(payment_record)
    
    return {
        "success": True,
        "payment": payment_record,
        "message": payment_record["message"]
    }

@router.get("/payment/history")
async def get_payment_history():
    return {"payments": payment_history, "total": len(payment_history)}
