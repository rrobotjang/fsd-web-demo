interface QwenPanelProps {
  narration: string
  situation: string
  confidence: number
  isSpeaking: boolean
  isListening: boolean
  ttsSupported: boolean
  sttSupported: boolean
  voiceStatus: string
  onSpeak: () => void
  onToggleListening: () => void
}

export default function QwenPanel({
  narration,
  situation,
  confidence,
  isSpeaking,
  isListening,
  ttsSupported,
  sttSupported,
  voiceStatus,
  onSpeak,
  onToggleListening,
}: QwenPanelProps) {
  const situationColors: Record<string, string> = {
    normal_driving: 'bg-green-900',
    pedestrian_warning: 'bg-yellow-900',
    traffic_sign_detected: 'bg-blue-900',
    unknown: 'bg-gray-900'
  }

  return (
    <div className="bg-gray-800 rounded-lg p-4 h-full">
      <h3 className="text-lg font-bold mb-3 text-blue-400">AI Narration</h3>
      
      <div className={`p-3 rounded mb-3 ${situationColors[situation] || 'bg-gray-900'}`}>
        <p className="text-sm text-gray-400">Situation: {situation}</p>
        <p className="text-xs text-gray-500">Confidence: {(confidence * 100).toFixed(0)}%</p>
      </div>
      
      <div className="bg-gray-900 rounded p-3 min-h-[100px]">
        <p className="text-white leading-relaxed">
          {narration || 'Waiting for AI analysis...'}
        </p>
      </div>

      <div className="mt-3 grid grid-cols-2 gap-2">
        <button
          type="button"
          onClick={onSpeak}
          disabled={!narration || !ttsSupported}
          className="rounded bg-blue-600 px-3 py-2 text-sm font-semibold hover:bg-blue-500 disabled:cursor-not-allowed disabled:bg-gray-700 disabled:text-gray-500"
        >
          {isSpeaking ? 'Stop voice' : 'Read aloud'}
        </button>
        <button
          type="button"
          onClick={onToggleListening}
          disabled={!sttSupported}
          className={`rounded px-3 py-2 text-sm font-semibold disabled:cursor-not-allowed disabled:bg-gray-700 disabled:text-gray-500 ${
            isListening ? 'bg-red-600 hover:bg-red-500' : 'bg-green-600 hover:bg-green-500'
          }`}
        >
          {isListening ? 'Stop listening' : 'Voice command'}
        </button>
      </div>
      {!ttsSupported && (
        <p className="mt-2 text-xs text-gray-500">Text-to-speech is unavailable in this browser.</p>
      )}
      {!sttSupported && (
        <p className="mt-2 text-xs text-gray-500">Voice commands require Chrome or Edge microphone support.</p>
      )}
      {voiceStatus && (
        <p className={`mt-2 text-xs ${isListening ? 'text-green-400' : 'text-yellow-400'}`}>{voiceStatus}</p>
      )}
    </div>
  )
}
