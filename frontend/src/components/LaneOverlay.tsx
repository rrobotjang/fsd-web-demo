import { useRef, useEffect } from 'react'

interface Lane {
  points: number[][]
}

interface LaneOverlayProps {
  lanes: Lane[]
  isVisible: boolean
}

export default function LaneOverlay({ lanes, isVisible }: LaneOverlayProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)

  useEffect(() => {
    if (!isVisible) return
    
    const canvas = canvasRef.current
    if (!canvas) return
    
    const ctx = canvas.getContext('2d')
    if (!ctx) return

    ctx.clearRect(0, 0, canvas.width, canvas.height)
    
    const laneColors = ['#FF0000', '#00FF00', '#0000FF']
    
    lanes.forEach((lane, idx) => {
      if (lane.points.length < 2) return
      
      ctx.strokeStyle = laneColors[idx % laneColors.length]
      ctx.lineWidth = 3
      ctx.beginPath()
      
      const [firstX, firstY] = lane.points[0]
      ctx.moveTo(firstX, firstY)
      
      for (let i = 1; i < lane.points.length; i++) {
        const [x, y] = lane.points[i]
        ctx.lineTo(x, y)
      }
      
      ctx.stroke()
    })
  }, [lanes, isVisible])

  if (!isVisible) return null

  return (
    <canvas
      ref={canvasRef}
      width={640}
      height={480}
      className="absolute inset-0 pointer-events-none"
    />
  )
}
