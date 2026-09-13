import { useState, useCallback } from 'react'

interface WebSocketMessage {
  frame_id: number
  detections: Array<{
    class: string
    confidence: number
    bbox: number[]
  }>
  lanes: Array<{ points: number[][] }>
  narration: string
  timestamp: number
}

export function useWebSocket(url: string) {
  const [isConnected, setIsConnected] = useState(false)
  const [lastMessage, setLastMessage] = useState<WebSocketMessage | null>(null)
  const [error, setError] = useState<string | null>(null)

  const connect = useCallback(() => {
    const ws = new WebSocket(url)
    
    ws.onopen = () => {
      setIsConnected(true)
      setError(null)
    }
    
    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data) as WebSocketMessage
        setLastMessage(data)
      } catch (e) {
        setError('Failed to parse message')
      }
    }
    
    ws.onerror = () => {
      setError('WebSocket error')
    }
    
    ws.onclose = () => {
      setIsConnected(false)
    }
    
    return ws
  }, [url])

  const sendMessage = useCallback((ws: WebSocket, data: unknown) => {
    if (ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify(data))
    }
  }, [])

  return { isConnected, lastMessage, error, connect, sendMessage }
}
