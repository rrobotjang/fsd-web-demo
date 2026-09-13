import { useState, useRef, useCallback, useEffect } from 'react'
import DrivingView from './DrivingView'
import LaneOverlay from './LaneOverlay'
import QwenPanel from './QwenPanel'
import PaymentSimulator from './PaymentSimulator'

interface Detection {
  class: string
  confidence: number
  bbox: number[]
}

interface Lane {
  points: number[][]
}

interface Payment {
  id: number
  scenario: string
  scenario_name: string
  amount: number
  status: string
  message: string
}

interface DemoState {
  isRunning: boolean
  timeElapsed: number
  phase: 'normal' | 'toll' | 'parking' | 'fuel' | 'complete'
}

interface SpeechRecognitionEventLike {
  results: { [index: number]: { [index: number]: { transcript: string } } }
}

interface SpeechRecognitionErrorEventLike {
  error?: string
}

interface SpeechRecognitionLike {
  continuous: boolean
  interimResults: boolean
  lang: string
  start: () => void
  stop: () => void
  onresult: ((event: SpeechRecognitionEventLike) => void) | null
  onend: (() => void) | null
  onerror: ((event: SpeechRecognitionErrorEventLike) => void) | null
}

type SpeechRecognitionConstructor = new () => SpeechRecognitionLike

const PHASES = [
  { name: 'normal', duration: 30, label: 'Normal Driving' },
  { name: 'toll', duration: 15, label: 'Toll Gate' },
  { name: 'parking', duration: 15, label: 'Parking' },
  { name: 'fuel', duration: 15, label: 'Fuel Station' },
  { name: 'complete', duration: 15, label: 'Destination' }
]

export default function Dashboard() {
  const [detections, setDetections] = useState<Detection[]>([])
  const [lanes, setLanes] = useState<Lane[]>([])
  const [narration, setNarration] = useState('')
  const [situation, setSituation] = useState('normal_driving')
  const [confidence, setConfidence] = useState(0)
  const [payments, setPayments] = useState<Payment[]>([])
  const [showLanes, setShowLanes] = useState(true)
  const [frame, setFrame] = useState('')
  const [isSpeaking, setIsSpeaking] = useState(false)
  const [isListening, setIsListening] = useState(false)
  const [voiceStatus, setVoiceStatus] = useState('')
  
  const [demoState, setDemoState] = useState<DemoState>({
    isRunning: false,
    timeElapsed: 0,
    phase: 'normal'
  })
  const [cameraMode, setCameraMode] = useState(false)
  const [cameraActive, setCameraActive] = useState(false)
  const [cameraError, setCameraError] = useState('')

  const wsRef = useRef<WebSocket | null>(null)
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null)
  const autoPaymentPhaseRef = useRef<DemoState['phase'] | null>(null)
  const recognitionRef = useRef<SpeechRecognitionLike | null>(null)
  const lastSpokenNarrationRef = useRef('')
  const videoRef = useRef<HTMLVideoElement>(null)
  const camStreamRef = useRef<MediaStream | null>(null)
  const captureTimerRef = useRef<ReturnType<typeof setInterval> | null>(null)
  const pendingCaptureRef = useRef(false)

  const ttsSupported = typeof window !== 'undefined' && 'speechSynthesis' in window
  const sttSupported = typeof window !== 'undefined' && (
    'SpeechRecognition' in window || 'webkitSpeechRecognition' in window
  )

  const speakNarration = useCallback((text: string) => {
    if (!('speechSynthesis' in window) || !text) return
    window.speechSynthesis.cancel()
    const utterance = new SpeechSynthesisUtterance(text)
    utterance.onstart = () => setIsSpeaking(true)
    utterance.onend = () => setIsSpeaking(false)
    utterance.onerror = () => setIsSpeaking(false)
    window.speechSynthesis.speak(utterance)
  }, [])

  useEffect(() => {
    if (!cameraActive || !videoRef.current || !camStreamRef.current) return
    videoRef.current.srcObject = camStreamRef.current
    void videoRef.current.play().catch(() => setCameraError('Camera preview could not start.'))
  }, [cameraActive])

  const toggleCamera = useCallback(async () => {
    if (cameraMode) {
      if (captureTimerRef.current) {
        clearInterval(captureTimerRef.current)
        captureTimerRef.current = null
      }
      camStreamRef.current?.getTracks().forEach(track => track.stop())
      camStreamRef.current = null
      if (videoRef.current) videoRef.current.srcObject = null
      pendingCaptureRef.current = false
      setCameraActive(false)
      setCameraMode(false)
      return
    }
    if (!navigator.mediaDevices?.getUserMedia) {
      setCameraError('Camera API is not available in this browser. Use Chrome or Edge.')
      return
    }
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 640 }, height: { ideal: 480 } }
      })
      camStreamRef.current = stream
      setCameraError('')
      setCameraActive(true)
      setCameraMode(true)
    } catch (error) {
      const name = error instanceof DOMException ? error.name : ''
      if (name === 'NotAllowedError' || name === 'PermissionDeniedError') {
        setCameraError('Camera blocked. Allow camera access for localhost, then try again.')
      } else if (name === 'NotFoundError' || name === 'OverconstrainedError') {
        setCameraError('No camera was found. Connect a webcam and try again.')
      } else {
        setCameraError(`Camera could not be opened: ${name || 'unknown error'}.`)
      }
    }
  }, [cameraMode])

  const startCaptureLoop = useCallback((ws: WebSocket) => {
    if (captureTimerRef.current) clearInterval(captureTimerRef.current)
    pendingCaptureRef.current = false
    captureTimerRef.current = setInterval(() => {
      const video = videoRef.current
      if (!video || video.readyState < 2 || ws.readyState !== WebSocket.OPEN) return
      if (pendingCaptureRef.current) return
      const canvas = document.createElement('canvas')
      canvas.width = 640
      canvas.height = 480
      const ctx = canvas.getContext('2d')
      if (!ctx) return
      ctx.drawImage(video, 0, 0, 640, 480)
      const jpeg = canvas.toDataURL('image/jpeg', 0.8)
      pendingCaptureRef.current = true
      ws.send(JSON.stringify({ image: jpeg.split(',')[1] }))
    }, 250)
  }, [])

  const handleVoiceCommand = useCallback((transcript: string) => {
    const command = transcript.toLowerCase()
    if (command.includes('start') || command.includes('begin')) {
      if (!demoState.isRunning) startDemo()
      setVoiceStatus('Command received: starting demo.')
    } else if (command.includes('stop') || command.includes('end')) {
      if (demoState.isRunning) stopDemo()
      setVoiceStatus('Command received: stopping demo.')
    } else if (command.includes('lane')) {
      setShowLanes(current => !current)
      setVoiceStatus('Command received: toggling lanes.')
    } else {
      setVoiceStatus(`Heard: "${transcript}". Try start, stop, or toggle lanes.`)
    }
  }, [demoState.isRunning, cameraMode])

  const toggleListening = useCallback(async () => {
    if (isListening) {
      recognitionRef.current?.stop()
      setIsListening(false)
      return
    }

    const Recognition = (window as Window & {
      SpeechRecognition?: SpeechRecognitionConstructor
      webkitSpeechRecognition?: SpeechRecognitionConstructor
    }).SpeechRecognition || (window as Window & {
      webkitSpeechRecognition?: SpeechRecognitionConstructor
    }).webkitSpeechRecognition
    if (!Recognition) {
      setVoiceStatus('Voice commands are not supported in this browser. Use Chrome or Edge.')
      return
    }

    if (navigator.mediaDevices?.getUserMedia) {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
        stream.getTracks().forEach(track => track.stop())
      } catch (error) {
        const name = error instanceof DOMException ? error.name : ''
        if (name === 'NotAllowedError' || name === 'PermissionDeniedError') {
          setVoiceStatus('Microphone blocked. Allow microphone access for localhost, then try again.')
        } else if (name === 'NotFoundError') {
          setVoiceStatus('No microphone was found. Connect or enable a microphone.')
        } else {
          setVoiceStatus('Microphone could not be opened. Check browser and macOS permissions.')
        }
        return
      }
    }

    const recognition = new Recognition()
    recognition.lang = 'en-US'
    recognition.continuous = false
    recognition.interimResults = false
    recognition.onresult = event => {
      const transcript = event.results[0][0].transcript.trim()
      handleVoiceCommand(transcript)
    }
    recognition.onend = () => setIsListening(false)
    recognition.onerror = event => {
      setIsListening(false)
      if (event.error === 'not-allowed' || event.error === 'service-not-allowed') {
        setVoiceStatus('Microphone blocked. Allow microphone access for localhost, then try again.')
      } else if (event.error === 'audio-capture') {
        setVoiceStatus('No microphone is available.')
      } else if (event.error === 'network') {
        setVoiceStatus('Speech recognition needs a network connection in this browser.')
      } else {
        setVoiceStatus(`Voice recognition failed: ${event.error || 'unknown error'}.`)
      }
    }
    recognitionRef.current = recognition
    recognition.start()
    setIsListening(true)
    setVoiceStatus('Listening: say start, stop, or toggle lanes.')
  }, [handleVoiceCommand, isListening])

  const connectWebSocket = useCallback(() => {
    const ws = new WebSocket('ws://localhost:8000/ws/stream')
    
    ws.onopen = () => {
      console.log('WebSocket connected')
      if (!cameraMode) {
        ws.send(JSON.stringify({ start: true }))
      }
    }
    
    ws.onmessage = (event) => {
      pendingCaptureRef.current = false
      try {
        const data = JSON.parse(event.data)
        setDetections(data.detections || [])
        setLanes(data.lanes || [])
        setNarration(data.narration || '')
        if (data.situation) setSituation(data.situation)
        if (data.confidence) setConfidence(data.confidence)
        if (data.frame) setFrame(data.frame)
      } catch (e) {
        console.error('Failed to parse message:', e)
      }
    }
    
    ws.onerror = (error) => {
      console.error('WebSocket error:', error)
    }
    
    ws.onclose = () => {
      console.log('WebSocket disconnected')
    }
    
    wsRef.current = ws
    return ws
  }, [cameraMode])

  const startDemo = () => {
    const ws = connectWebSocket()
    
    if (cameraMode) {
      startCaptureLoop(ws)
    }
    
    setDemoState({
      isRunning: true,
      timeElapsed: 0,
      phase: 'normal'
    })
    autoPaymentPhaseRef.current = null
    
    timerRef.current = setInterval(() => {
      setDemoState(prev => {
        const newTime = prev.timeElapsed + 1
        
        let newPhase: DemoState['phase'] = 'normal'
        if (newTime >= 75) newPhase = 'complete'
        else if (newTime >= 60) newPhase = 'fuel'
        else if (newTime >= 45) newPhase = 'parking'
        else if (newTime >= 30) newPhase = 'toll'
        
        if (newTime >= 90) {
          if (timerRef.current) clearInterval(timerRef.current)
          ws.close()
          return { isRunning: false, timeElapsed: 90, phase: 'complete' }
        }
        
        return { ...prev, timeElapsed: newTime, phase: newPhase }
      })
    }, 1000)
  }

  const stopDemo = () => {
    if (timerRef.current) clearInterval(timerRef.current)
    if (captureTimerRef.current) {
      clearInterval(captureTimerRef.current)
      captureTimerRef.current = null
    }
    pendingCaptureRef.current = false
    if (wsRef.current) wsRef.current.close()
    
    setDemoState({
      isRunning: false,
      timeElapsed: 0,
      phase: 'normal'
    })
    autoPaymentPhaseRef.current = null
  }

  useEffect(() => {
    if (!narration || narration === lastSpokenNarrationRef.current) return
    lastSpokenNarrationRef.current = narration
  }, [narration])

  useEffect(() => () => {
    recognitionRef.current?.stop()
    window.speechSynthesis?.cancel()
    if (captureTimerRef.current) clearInterval(captureTimerRef.current)
    camStreamRef.current?.getTracks().forEach(track => track.stop())
  }, [])

  const handlePayment = async (scenario: string) => {
    try {
      const response = await fetch('http://localhost:8000/api/payment', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ scenario })
      })
      const data = await response.json()
      if (data.success && data.payment) {
        setPayments(prev => [...prev, data.payment])
      }
    } catch (e) {
      console.error('Payment error:', e)
    }
  }

  useEffect(() => {
    const phase = demoState.phase
    if (!demoState.isRunning || !['toll', 'parking', 'fuel'].includes(phase)) return
    if (autoPaymentPhaseRef.current === phase) return

    autoPaymentPhaseRef.current = phase
    void handlePayment(phase)
  }, [demoState.isRunning, demoState.phase])

  useEffect(() => {
    if (!demoState.isRunning) return
    const announcements: Partial<Record<DemoState['phase'], string>> = {
      normal: 'Normal driving has started.',
      toll: 'Toll gate ahead. Toll payment is being processed.',
      parking: 'Parking area ahead. Parking payment is being processed.',
      fuel: 'Fuel station ahead. Fuel payment is being processed.',
      complete: 'Destination reached. The driving demo is complete.',
    }
    const announcement = announcements[demoState.phase]
    if (announcement) speakNarration(announcement)
  }, [demoState.isRunning, demoState.phase, speakNarration])

  const currentPhase = PHASES.find(p => p.name === demoState.phase)
  const progress = (demoState.timeElapsed / 90) * 100

  return (
    <div className="min-h-screen bg-gray-900 text-white p-6">
      <div className="max-w-7xl mx-auto">
        <div className="flex justify-between items-center mb-6">
          <h1 className="text-3xl font-bold">FSD Web Demo</h1>
          <div className="flex items-center gap-4">
            <span className="text-gray-400">
              {demoState.timeElapsed}s / 90s
            </span>
            {cameraActive && (
              <video
                ref={videoRef}
                playsInline
                muted
                className="w-28 h-24 rounded-lg border border-gray-700 object-cover bg-black"
              />
            )}
            <button
              onClick={toggleCamera}
              className={`px-4 py-2 rounded-lg font-bold ${
                cameraMode
                  ? 'bg-gray-600 hover:bg-gray-700'
                  : 'bg-indigo-600 hover:bg-indigo-700'
              }`}
            >
              {cameraMode ? 'Stop Camera' : 'Use Camera'}
            </button>
            <button
              onClick={demoState.isRunning ? stopDemo : startDemo}
              className={`px-6 py-2 rounded-lg font-bold ${
                demoState.isRunning
                  ? 'bg-red-600 hover:bg-red-700'
                  : 'bg-green-600 hover:bg-green-700'
              }`}
            >
              {demoState.isRunning ? 'Stop' : 'Start Demo'}
            </button>
          </div>
        </div>

        <div className="mb-4 bg-gray-800 rounded-lg p-3">
          <div className="flex justify-between text-sm mb-2">
            <span>
              {currentPhase?.label || 'Ready'}
              <span className="ml-2 text-xs text-gray-400">
                Source: {cameraMode ? 'Webcam' : 'Demo'}
              </span>
            </span>
            <span>{Math.round(progress)}%</span>
          </div>
          {cameraError && (
            <div className="mt-2 text-red-400 text-sm">{cameraError}</div>
          )}
          <div className="w-full bg-gray-700 rounded-full h-2">
            <div
              className="bg-blue-500 h-2 rounded-full transition-all"
              style={{ width: `${progress}%` }}
            />
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          <div className="lg:col-span-2">
            <div className="relative">
              <DrivingView detections={detections} isStreaming={demoState.isRunning} frame={frame} />
              <LaneOverlay lanes={lanes} isVisible={showLanes} />
            </div>
            <div className="mt-2 flex gap-2">
              <button
                onClick={() => setShowLanes(!showLanes)}
                className={`px-4 py-1 rounded text-sm ${
                  showLanes ? 'bg-blue-600' : 'bg-gray-600'
                }`}
              >
                Lanes: {showLanes ? 'ON' : 'OFF'}
              </button>
            </div>
          </div>
          
          <div className="flex flex-col gap-4">
            <QwenPanel
              narration={narration}
              situation={situation}
              confidence={confidence}
              isSpeaking={isSpeaking}
              isListening={isListening}
              ttsSupported={ttsSupported}
              sttSupported={sttSupported}
              voiceStatus={voiceStatus}
              onSpeak={() => isSpeaking ? window.speechSynthesis.cancel() : speakNarration(narration)}
              onToggleListening={toggleListening}
            />
            <PaymentSimulator
              onPayment={handlePayment}
              payments={payments}
            />
          </div>
        </div>

        <div className="mt-6 grid grid-cols-4 gap-4">
          {['Normal', 'Toll Gate', 'Parking', 'Fuel Station'].map((phase, idx) => (
            <div
              key={phase}
              className={`p-3 rounded-lg text-center ${
                demoState.phase === ['normal', 'toll', 'parking', 'fuel'][idx]
                  ? 'bg-blue-600'
                  : 'bg-gray-800'
              }`}
            >
              <div className="text-sm">{phase}</div>
              <div className="text-xs text-gray-400">
                {idx * 15}-{(idx + 1) * 15}s
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}
