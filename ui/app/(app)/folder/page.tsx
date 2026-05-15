'use client'

import { useCallback, useEffect, useRef, useState } from 'react'
import Image from 'next/image'
import {
  FolderOpenIcon,
  PlayIcon,
  CheckCircle2Icon,
  ImageIcon,
} from 'lucide-react'
import { api, FolderFileInfo, FolderSessionInfo } from '@/lib/api'
import { useDocumentMutations } from '@/lib/query/mutations'
import { useOperationStore } from '@/lib/stores/operationStore'
import { Button } from '@/components/ui/button'

export default function FolderPage() {
  const [session, setSession] = useState<FolderSessionInfo | null>(null)
  const [loading, setLoading] = useState(false)

  const { openFolderSession, startFolderPipeline } = useDocumentMutations()
  const operation = useOperationStore((s) => s.operation)

  const isRunning =
    operation?.type === 'process-all' && !operation.cancelRequested
  const current = operation?.current ?? 0
  const total = operation?.total ?? 1
  const percent = total > 0 ? Math.round((current / total) * 100) : 0

  const handleOpenFolder = useCallback(async () => {
    setLoading(true)
    try {
      const s = await openFolderSession()
      if (s) setSession(s)
    } finally {
      setLoading(false)
    }
  }, [openFolderSession])

  const handleProcessAll = useCallback(async () => {
    await startFolderPipeline()
  }, [startFolderPipeline])

  // Refresh session after pipeline completes to show updated hasResult flags.
  useEffect(() => {
    if (!operation && session) {
      api
        .getFolderSession()
        .then((s) => {
          if (s) setSession(s)
        })
        .catch(() => {})
    }
  }, [operation])

  return (
    <div className='flex h-full flex-col overflow-hidden'>
      {/* Toolbar */}
      <div className='border-border flex h-12 items-center gap-3 border-b px-4'>
        <Button
          size='sm'
          variant='outline'
          onClick={handleOpenFolder}
          disabled={loading || isRunning}
        >
          <FolderOpenIcon className='mr-1.5 size-4' />
          Open Folder
        </Button>

        {session && (
          <>
            <span className='text-muted-foreground max-w-xs truncate text-xs'>
              {session.root}
            </span>
            <span className='text-muted-foreground text-xs'>
              {session.files.length} images
            </span>
            <div className='flex-1' />
            <Button
              size='sm'
              onClick={handleProcessAll}
              disabled={isRunning || session.files.length === 0}
            >
              <PlayIcon className='mr-1.5 size-4' />
              {isRunning ? 'Processing…' : 'Process All'}
            </Button>
          </>
        )}
      </div>

      {/* Progress bar */}
      {isRunning && (
        <div className='border-border border-b px-4 py-2'>
          <div className='mb-1 flex items-center justify-between text-xs'>
            <span className='text-muted-foreground'>
              {operation?.step ?? ''} — {current} / {total}
            </span>
            <span className='font-medium'>{percent}%</span>
          </div>
          <div className='bg-muted h-1.5 w-full overflow-hidden rounded-full'>
            <div
              className='bg-primary h-full transition-all'
              style={{ width: `${percent}%` }}
            />
          </div>
        </div>
      )}

      {/* File grid */}
      {session ? (
        <div className='grid flex-1 grid-cols-[repeat(auto-fill,minmax(160px,1fr))] gap-3 overflow-y-auto p-4'>
          {session.files.map((file) => (
            <FolderFileCard
              key={file.index}
              file={file}
              isRunning={isRunning}
            />
          ))}
        </div>
      ) : (
        <div className='flex flex-1 flex-col items-center justify-center gap-3 text-center'>
          <FolderOpenIcon className='text-muted-foreground size-14 opacity-30' />
          <p className='text-muted-foreground text-sm'>
            Open a folder to translate all images inside it.
            <br />
            Results are saved to a <code>result/</code> subfolder automatically.
          </p>
          <Button
            variant='outline'
            onClick={handleOpenFolder}
            disabled={loading}
          >
            <FolderOpenIcon className='mr-1.5 size-4' />
            Open Folder
          </Button>
        </div>
      )}
    </div>
  )
}

function FolderFileCard({
  file,
  isRunning,
}: {
  file: FolderFileInfo
  isRunning: boolean
}) {
  const ref = useRef<HTMLDivElement>(null)
  const [visible, setVisible] = useState(false)
  const [showResult, setShowResult] = useState(file.hasResult)

  useEffect(() => {
    setShowResult(file.hasResult)
  }, [file.hasResult])

  // Lazy-load: only fetch when in viewport.
  useEffect(() => {
    const el = ref.current
    if (!el) return
    const obs = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting) setVisible(true)
      },
      { rootMargin: '200px' },
    )
    obs.observe(el)
    return () => obs.disconnect()
  }, [])

  const imgSrc = showResult
    ? api.getFolderResultUrl(file.index)
    : api.getFolderImageUrl(file.index)

  return (
    <div
      ref={ref}
      className='bg-muted/40 group relative overflow-hidden rounded-lg border'
    >
      {/* Image */}
      <div className='relative aspect-[3/4] w-full overflow-hidden bg-black/5'>
        {visible ? (
          <Image
            src={imgSrc}
            alt={file.name}
            fill
            className='object-contain'
            unoptimized
          />
        ) : (
          <div className='flex h-full items-center justify-center'>
            <ImageIcon className='text-muted-foreground size-8 opacity-30' />
          </div>
        )}

        {/* Result badge */}
        {file.hasResult && (
          <div className='absolute top-1.5 right-1.5'>
            <CheckCircle2Icon className='size-4 fill-green-500 text-white drop-shadow' />
          </div>
        )}

        {/* Toggle result/original button */}
        {file.hasResult && visible && (
          <button
            className='bg-background/80 absolute right-1.5 bottom-1.5 rounded px-1.5 py-0.5 text-[10px] font-medium opacity-0 backdrop-blur-sm transition-opacity group-hover:opacity-100'
            onClick={() => setShowResult((p) => !p)}
          >
            {showResult ? 'Original' : 'Result'}
          </button>
        )}
      </div>

      {/* Name */}
      <div className='px-2 py-1.5'>
        <p className='truncate text-xs font-medium'>{file.name}</p>
        <p className='text-muted-foreground text-[10px]'>
          {file.width}×{file.height}
        </p>
      </div>
    </div>
  )
}
