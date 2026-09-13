import { useState } from 'react'

interface Payment {
  id: number
  scenario: string
  scenario_name: string
  amount: number
  status: string
  message: string
}

interface PaymentSimulatorProps {
  onPayment: (scenario: string) => void
  payments: Payment[]
}

export default function PaymentSimulator({ onPayment, payments }: PaymentSimulatorProps) {
  const [selectedScenario, setSelectedScenario] = useState<string | null>(null)
  const [isProcessing, setIsProcessing] = useState(false)

  const scenarios = [
    { id: 'toll', name: 'Toll Gate', icon: ' toll' },
    { id: 'parking', name: 'Parking', icon: '🅿️' },
    { id: 'fuel', name: 'Fuel Station', icon: '⛽' }
  ]

  const handlePayment = async () => {
    if (!selectedScenario) return
    
    setIsProcessing(true)
    onPayment(selectedScenario)
    
    setTimeout(() => {
      setIsProcessing(false)
      setSelectedScenario(null)
    }, 1500)
  }

  return (
    <div className="bg-gray-800 rounded-lg p-4 h-full">
      <h3 className="text-lg font-bold mb-3 text-green-400">Payment Simulation</h3>
      
      <div className="grid grid-cols-3 gap-2 mb-4">
        {scenarios.map(scenario => (
          <button
            key={scenario.id}
            onClick={() => setSelectedScenario(scenario.id)}
            className={`p-3 rounded-lg transition-all ${
              selectedScenario === scenario.id
                ? 'bg-green-600 text-white'
                : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
            }`}
          >
            <div className="text-2xl mb-1">{scenario.icon}</div>
            <div className="text-xs">{scenario.name}</div>
          </button>
        ))}
      </div>
      
      <button
        onClick={handlePayment}
        disabled={!selectedScenario || isProcessing}
        className={`w-full py-2 rounded-lg font-bold transition-all ${
          selectedScenario && !isProcessing
            ? 'bg-green-500 hover:bg-green-600 text-white'
            : 'bg-gray-600 text-gray-400 cursor-not-allowed'
        }`}
      >
        {isProcessing ? 'Processing...' : 'Pay Now'}
      </button>
      
      {payments.length > 0 && (
        <div className="mt-4 max-h-[150px] overflow-y-auto">
          <h4 className="text-sm font-bold text-gray-400 mb-2">Payment History</h4>
          {payments.slice(-3).reverse().map(payment => (
            <div key={payment.id} className="text-xs bg-gray-900 rounded p-2 mb-1">
              <span className="text-green-400">${payment.amount.toFixed(2)}</span>
              <span className="text-gray-500 ml-2">{payment.scenario_name}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
