'use client'

import { useEditorUiStore, type ScanPanel } from '@/lib/stores/editorUiStore'

// 8 cycling colors for panel outlines (P0=red, P1=orange, P2=yellow, P3=green,
// P4=sky-blue, P5=blue, P6=violet, P7=pink)
const PANEL_COLORS = [
  { border: '#ef4444', bg: 'rgba(239,68,68,0.08)', label: 'bg-red-600' },
  { border: '#f97316', bg: 'rgba(249,115,22,0.08)', label: 'bg-orange-600' },
  { border: '#eab308', bg: 'rgba(234,179,8,0.08)', label: 'bg-yellow-500' },
  { border: '#22c55e', bg: 'rgba(34,197,94,0.08)', label: 'bg-green-600' },
  { border: '#0ea5e9', bg: 'rgba(14,165,233,0.08)', label: 'bg-sky-600' },
  { border: '#3b82f6', bg: 'rgba(59,130,246,0.08)', label: 'bg-blue-600' },
  { border: '#8b5cf6', bg: 'rgba(139,92,246,0.08)', label: 'bg-violet-600' },
  { border: '#ec4899', bg: 'rgba(236,72,153,0.08)', label: 'bg-pink-600' },
]

type PanelAnnotationsProps = {
  style?: React.CSSProperties
}

export function PanelAnnotations({ style }: PanelAnnotationsProps) {
  const scanPanels = useEditorUiStore((state) => state.scanPanels)
  const scale = useEditorUiStore((state) => state.scale)
  const scaleRatio = scale / 100

  if (scanPanels.length === 0) return null

  return (
    <div className='pointer-events-none absolute inset-0' style={style}>
      {scanPanels.map((panel, index) => (
        <PanelBox
          key={index}
          panel={panel}
          index={index}
          scaleRatio={scaleRatio}
        />
      ))}
    </div>
  )
}

function PanelBox({
  panel,
  index,
  scaleRatio,
}: {
  panel: ScanPanel
  index: number
  scaleRatio: number
}) {
  const color = PANEL_COLORS[index % PANEL_COLORS.length]

  return (
    <div
      className='absolute'
      style={{
        left: panel.x * scaleRatio,
        top: panel.y * scaleRatio,
        width: panel.width * scaleRatio,
        height: panel.height * scaleRatio,
      }}
    >
      <div
        className='absolute inset-0 rounded border-2'
        style={{ borderColor: color.border, backgroundColor: color.bg }}
      />
      <div
        className={`absolute -top-5 left-0 rounded px-1 py-0.5 text-[10px] font-semibold text-white ${color.label}`}
      >
        P{index}
      </div>
    </div>
  )
}
