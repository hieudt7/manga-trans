'use client'

import { useCurrentDocumentState } from '@/lib/query/hooks'
import { useEditorUiStore } from '@/lib/stores/editorUiStore'
import type { BalloonDetection } from '@/types'

type BalloonAnnotationsProps = {
  style?: React.CSSProperties
}

export function BalloonAnnotations({ style }: BalloonAnnotationsProps) {
  const { currentDocument } = useCurrentDocumentState()
  const scale = useEditorUiStore((state) => state.scale)
  const scaleRatio = scale / 100
  const balloons = currentDocument?.balloons ?? []

  if (balloons.length === 0) return null

  return (
    <div className='pointer-events-none absolute inset-0' style={style}>
      {balloons.map((balloon, index) => (
        <BalloonBox key={index} balloon={balloon} scaleRatio={scaleRatio} />
      ))}
    </div>
  )
}

function BalloonBox({
  balloon,
  scaleRatio,
}: {
  balloon: BalloonDetection
  scaleRatio: number
}) {
  return (
    <div
      className='absolute'
      style={{
        left: balloon.x * scaleRatio,
        top: balloon.y * scaleRatio,
        width: balloon.width * scaleRatio,
        height: balloon.height * scaleRatio,
      }}
    >
      <div className='absolute inset-0 rounded border-2 border-blue-500 bg-blue-500/10' />
      <div className='absolute -top-5 left-0 rounded bg-blue-600 px-1 py-0.5 text-[10px] font-semibold text-white'>
        balloon {balloon.score.toFixed(2)}
      </div>
    </div>
  )
}
