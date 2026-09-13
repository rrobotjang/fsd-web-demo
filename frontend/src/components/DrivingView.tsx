import { useRef, useEffect } from 'react'

interface Detection {
  class: string
  confidence: number
  bbox: number[]
}

interface DrivingViewProps {
  detections: Detection[]
  isStreaming: boolean
  frame?: string
}

const CLASS_COLORS: Record<string, string> = {
  car: '#00FF00',
  truck: '#FFA500',
  bus: '#FFA500',
  person: '#FFFF00',
  pedestrian: '#FFFF00',
  bicycle: '#00FFFF',
  motorcycle: '#00FFFF',
  'traffic light': '#FF00FF',
  'stop sign': '#FF00FF'
}

export default function DrivingView({ detections, isStreaming, frame }: DrivingViewProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return

    const ctx = canvas.getContext('2d')
    if (!ctx) return

    const drawDetections = () => {
      detections.forEach(det => {
        const [x1, y1, x2, y2] = det.bbox
        ctx.strokeStyle = CLASS_COLORS[det.class] || '#00FFFF'
        ctx.lineWidth = 2
        ctx.strokeRect(x1, y1, x2 - x1, y2 - y1)

        ctx.fillStyle = CLASS_COLORS[det.class] || '#00FFFF'
        ctx.font = '12px Arial'
        ctx.fillText(`${det.class} ${(det.confidence * 100).toFixed(0)}%`, x1, y1 - 5)
      })
    }

    if (frame) {
      const img = new Image()
      img.onload = () => {
        ctx.drawImage(img, 0, 0, canvas.width, canvas.height)
        drawDetections()
      }
      img.src = `data:image/jpeg;base64,${frame}`
    } else {
      ctx.fillStyle = '#1a1d23'
      ctx.fillRect(0, 0, canvas.width, canvas.height)
      drawDetections()
    }
  }, [detections, frame])

  return (
    <div className="relative">
      <canvas
        ref={canvasRef}
        width={640}
        height={480}
        className="rounded-lg border border-gray-700"
      />
      {!isStreaming && (
        <div className="absolute inset-0 flex items-center justify-center bg-black/50 rounded-lg">
          <span className="text-white text-xl">Press Start to Begin Demo</span>
        </div>
      )}
    </div>
  )
}