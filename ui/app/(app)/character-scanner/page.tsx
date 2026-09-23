'use client'

import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import Link from 'next/link'
import {
  ChevronLeftIcon,
  FolderOpenIcon,
  PlayIcon,
  StopCircleIcon,
  UserIcon,
  ImageIcon,
  XIcon,
  PlusIcon,
  Share2Icon,
  DownloadIcon,
  CheckIcon,
  SparklesIcon,
} from 'lucide-react'
import { useTranslation } from 'react-i18next'
import {
  Accordion,
  AccordionItem,
  AccordionTrigger,
  AccordionContent,
} from '@/components/ui/accordion'
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogFooter,
} from '@/components/ui/dialog'
import {
  Select,
  SelectTrigger,
  SelectValue,
  SelectContent,
  SelectItem,
} from '@/components/ui/select'
import { Tooltip, TooltipTrigger, TooltipContent } from '@/components/ui/tooltip'
import { Button } from '@/components/ui/button'
import { subscribeJobChanged } from '@/lib/backend'
import { useDocumentMutations } from '@/lib/query/mutations'
import {
  api,
  type CharacterScanResult,
  type RelationshipEdge,
  type ScannedCharacter,
  type FolderSessionInfo,
} from '@/lib/api'
import type { JobState } from '@/lib/protocol'

const GENDER_OPTIONS = ['male', 'female'] as const
const AGE_GROUP_OPTIONS = [
  'child',
  'teen',
  'young_adult',
  'adult',
  'middle_age',
  'elder',
] as const

const faceFileName = (path: string) => path.split('/').pop() ?? path

// ─── Avatar ─────────────────────────────────────────────────────────────────

function CharacterAvatar({ character }: { character: ScannedCharacter }) {
  const firstFace = character.faces[0]
  if (!firstFace) {
    return (
      <div className='bg-muted flex size-10 shrink-0 items-center justify-center rounded-full'>
        <UserIcon className='text-muted-foreground size-5' />
      </div>
    )
  }
  return (
    // eslint-disable-next-line @next/next/no-img-element
    <img
      src={api.getCharacterScanFaceUrl(character.id, faceFileName(firstFace))}
      alt={character.name}
      className='border-border size-10 shrink-0 rounded-full border object-cover'
    />
  )
}

// ─── Face Manager dialog ────────────────────────────────────────────────────

function FaceManagerDialog({
  character,
  open,
  onOpenChange,
  onSave,
}: {
  character: ScannedCharacter | null
  open: boolean
  onOpenChange: (open: boolean) => void
  onSave: (updated: ScannedCharacter) => void
}) {
  const { t } = useTranslation()
  const inputRef = useRef<HTMLInputElement>(null)
  const [name, setName] = useState('')
  const [gender, setGender] = useState('')
  const [ageGroup, setAgeGroup] = useState('')
  const [faces, setFaces] = useState<string[]>([])
  const [uploading, setUploading] = useState(false)

  useEffect(() => {
    if (character) {
      setName(character.name)
      setGender(character.gender ?? '')
      setAgeGroup(character.ageGroup ?? '')
      setFaces(character.faces)
    }
  }, [character])

  const handleAddFace = async (files: FileList | null) => {
    if (!character || !files?.length) return
    setUploading(true)
    try {
      for (const file of Array.from(files)) {
        const path = await api.addCharacterScanFace(character.id, file)
        setFaces((prev) => [...prev, path])
      }
    } catch (err) {
      console.error('Failed to add face', err)
    } finally {
      setUploading(false)
    }
  }

  if (!character) return null

  const handleSave = () => {
    onSave({
      ...character,
      name: name.trim() || character.name,
      gender: gender || null,
      ageGroup: ageGroup || null,
      faces,
    })
    onOpenChange(false)
  }

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className='max-h-[85vh] max-w-2xl overflow-y-auto'>
        <DialogHeader>
          <DialogTitle>{t('characterScanner.faceManager.title')}</DialogTitle>
        </DialogHeader>

        <div className='space-y-4'>
          <div className='space-y-1'>
            <label className='text-foreground text-sm'>
              {t('characterScanner.faceManager.name')}
            </label>
            <input
              value={name}
              onChange={(e) => setName(e.target.value)}
              className='border-border bg-card text-foreground focus:ring-primary w-full rounded-md border px-3 py-1.5 text-sm focus:ring-1 focus:outline-none'
            />
          </div>

          <div className='grid grid-cols-2 gap-3'>
            <div className='space-y-1'>
              <label className='text-foreground text-sm'>
                {t('characterScanner.faceManager.gender')}
              </label>
              <Select value={gender} onValueChange={setGender}>
                <SelectTrigger className='w-full'>
                  <SelectValue
                    placeholder={t('characterScanner.faceManager.unknown')}
                  />
                </SelectTrigger>
                <SelectContent>
                  {GENDER_OPTIONS.map((g) => (
                    <SelectItem key={g} value={g}>
                      {t(`characterScanner.genders.${g}`)}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
            <div className='space-y-1'>
              <label className='text-foreground text-sm'>
                {t('characterScanner.faceManager.ageGroup')}
              </label>
              <Select value={ageGroup} onValueChange={setAgeGroup}>
                <SelectTrigger className='w-full'>
                  <SelectValue
                    placeholder={t('characterScanner.faceManager.unknown')}
                  />
                </SelectTrigger>
                <SelectContent>
                  {AGE_GROUP_OPTIONS.map((a) => (
                    <SelectItem key={a} value={a}>
                      {t(`characterScanner.ageGroups.${a}`)}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
          </div>

          <div className='space-y-2'>
            <div className='flex items-center justify-between'>
              <label className='text-foreground text-sm'>
                {t('characterScanner.faceManager.faces', {
                  count: faces.length,
                })}
              </label>
              <button
                type='button'
                onClick={() => inputRef.current?.click()}
                disabled={uploading}
                className='text-primary hover:text-primary/80 inline-flex items-center gap-1 text-xs font-medium disabled:opacity-50'
              >
                <PlusIcon className='size-3' />
                {uploading
                  ? t('characterScanner.faceManager.uploading')
                  : t('characterScanner.faceManager.addFace')}
              </button>
              <input
                ref={inputRef}
                type='file'
                accept='image/*'
                multiple
                className='hidden'
                onChange={(e) => {
                  void handleAddFace(e.target.files)
                  e.target.value = ''
                }}
              />
            </div>
            <div className='grid grid-cols-4 gap-2 sm:grid-cols-6'>
              {faces.map((face) => (
                <div key={face} className='relative aspect-square'>
                  {/* eslint-disable-next-line @next/next/no-img-element */}
                  <img
                    src={api.getCharacterScanFaceUrl(
                      character.id,
                      faceFileName(face),
                    )}
                    alt={character.name}
                    className='border-border h-full w-full rounded-md border object-cover'
                  />
                  <button
                    type='button'
                    onClick={() =>
                      setFaces((prev) => prev.filter((f) => f !== face))
                    }
                    className='bg-background/80 hover:bg-background absolute -top-1 -right-1 rounded-full p-0.5'
                    aria-label={t('characterScanner.faceManager.deleteFace')}
                  >
                    <XIcon className='size-3' />
                  </button>
                </div>
              ))}
              {faces.length === 0 && (
                <p className='text-muted-foreground col-span-full py-4 text-center text-xs'>
                  {t('characterScanner.faceManager.noFaces')}
                </p>
              )}
            </div>
          </div>
        </div>

        <DialogFooter>
          <Button variant='outline' onClick={() => onOpenChange(false)}>
            {t('characterScanner.faceManager.cancel')}
          </Button>
          <Button onClick={handleSave}>
            {t('characterScanner.faceManager.save')}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}

// ─── Relationship row (inside accordion content) ───────────────────────────

function RelatedRow({
  character,
  label,
  description,
  coOccurrence,
  onClick,
}: {
  character: ScannedCharacter | undefined
  label: string | null
  description: string | null
  coOccurrence: number
  onClick: () => void
}) {
  const { t } = useTranslation()
  if (!character) return null
  const displayLabel =
    label ?? t('characterScanner.hubView.coOccurrence', { count: coOccurrence })
  return (
    <button
      type='button'
      onClick={onClick}
      className='hover:bg-accent flex w-full items-center gap-3 rounded-lg px-2 py-2 text-left transition'
    >
      <CharacterAvatar character={character} />
      <div className='min-w-0 flex-1'>
        <p className='text-foreground truncate text-sm font-medium'>
          {character.name}
          <span className='text-muted-foreground ml-2 text-xs font-normal'>
            {displayLabel}
          </span>
        </p>
        {description && (
          <p className='text-muted-foreground text-xs'>{description}</p>
        )}
      </div>
    </button>
  )
}

// ─── Character node (top-level accordion item) ─────────────────────────────

function CharacterNode({
  character,
  charactersById,
  relatedEntries,
  onManage,
  onSync,
  syncing,
  synced,
}: {
  character: ScannedCharacter
  charactersById: Map<string, ScannedCharacter>
  relatedEntries: RelationshipEdge[]
  onManage: (character: ScannedCharacter) => void
  onSync: (character: ScannedCharacter) => void
  syncing: boolean
  synced: boolean
}) {
  const { t } = useTranslation()
  return (
    <AccordionItem value={character.id} className='border-border border-b'>
      <div className='flex items-center gap-3 py-2'>
        <button
          type='button'
          onClick={() => onManage(character)}
          className='shrink-0'
          aria-label={t('characterScanner.faceManager.title')}
        >
          <CharacterAvatar character={character} />
        </button>
        <AccordionTrigger className='flex-1 py-0 hover:no-underline'>
          <div className='min-w-0 flex-1 text-left'>
            <p className='text-foreground truncate text-sm font-medium'>
              {character.name}
            </p>
            <div className='mt-0.5 flex flex-wrap gap-1'>
              {character.gender && (
                <span className='bg-accent text-foreground rounded-full px-2 py-0.5 text-[11px]'>
                  {t(`characterScanner.genders.${character.gender}`)}
                </span>
              )}
              {character.ageGroup && (
                <span className='bg-accent text-foreground rounded-full px-2 py-0.5 text-[11px]'>
                  {t(`characterScanner.ageGroups.${character.ageGroup}`)}
                </span>
              )}
            </div>
          </div>
        </AccordionTrigger>
        <button
          type='button'
          onClick={() => onSync(character)}
          disabled={syncing}
          className='text-muted-foreground hover:text-primary shrink-0 p-1 transition disabled:opacity-50'
          aria-label={t('characterScanner.syncToLibrary.button')}
          title={t('characterScanner.syncToLibrary.button')}
        >
          {synced ? (
            <CheckIcon className='size-4 text-green-600' />
          ) : (
            <Share2Icon className='size-4' />
          )}
        </button>
      </div>
      <AccordionContent>
        {character.traits.length > 0 && (
          <div className='space-y-1 px-2 pb-2'>
            <p className='text-muted-foreground text-[11px] font-medium tracking-wide uppercase'>
              {t('characterScanner.hubView.traitsLabel')}
            </p>
            {character.traits.map((trait, index) => (
              <p key={index} className='text-foreground text-xs leading-relaxed'>
                {trait}
              </p>
            ))}
          </div>
        )}
        {relatedEntries.length === 0 ? (
          <p className='text-muted-foreground px-2 text-xs'>
            {t('characterScanner.hubView.noRelations')}
          </p>
        ) : (
          <div className='space-y-1'>
            {relatedEntries.map((edge) => (
              <RelatedRow
                key={edge.characterId}
                character={charactersById.get(edge.characterId)}
                label={edge.label}
                description={edge.description}
                coOccurrence={edge.coOccurrence}
                onClick={() => {
                  const related = charactersById.get(edge.characterId)
                  if (related) onManage(related)
                }}
              />
            ))}
          </div>
        )}
      </AccordionContent>
    </AccordionItem>
  )
}

// ─── Page ───────────────────────────────────────────────────────────────────

export default function CharacterScannerPage() {
  const { t } = useTranslation()
  const { openFolderSession } = useDocumentMutations()

  const [session, setSession] = useState<FolderSessionInfo | null>(null)
  const [loadingFolder, setLoadingFolder] = useState(false)
  const [job, setJob] = useState<JobState | null>(null)
  const [result, setResult] = useState<CharacterScanResult | null>(null)
  const [loadingResult, setLoadingResult] = useState(false)
  const [managing, setManaging] = useState<ScannedCharacter | null>(null)
  const [dialogOpen, setDialogOpen] = useState(false)
  const [syncingId, setSyncingId] = useState<string | null>(null)
  const [syncedIds, setSyncedIds] = useState<Set<string>>(new Set())
  const [syncingAll, setSyncingAll] = useState(false)
  const [exporting, setExporting] = useState(false)
  const [generatingRelationships, setGeneratingRelationships] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const loadResult = useCallback(async () => {
    setLoadingResult(true)
    try {
      const r = await api.getCharacterScanResult()
      setResult(r)
    } catch {
      // no result yet — not an error the user needs to see
    } finally {
      setLoadingResult(false)
    }
  }, [])

  // Restore an already-open folder session (e.g. navigated from Folder Mode).
  useEffect(() => {
    void (async () => {
      try {
        const s = await api.getFolderSession()
        if (s) setSession(s)
      } catch {
        // no active session
      }
    })()
  }, [])

  useEffect(() => {
    if (session) void loadResult()
  }, [session?.root, loadResult])

  useEffect(() => {
    return subscribeJobChanged((j) => {
      if (j.kind !== 'character-scan-folder') return
      setJob(j)
      if (j.status === 'completed') void loadResult()
    })
  }, [loadResult])

  const isRunning = job?.status === 'running'

  const charactersById = useMemo(
    () => new Map((result?.characters ?? []).map((c) => [c.id, c])),
    [result],
  )

  const relatedByCharacter = useMemo(() => {
    const map = new Map<string, RelationshipEdge[]>()
    for (const node of result?.relationshipTree ?? []) {
      map.set(node.characterId, node.related)
    }
    return map
  }, [result])

  const hasUnlabeledEdges = useMemo(
    () =>
      (result?.relationshipTree ?? []).some((node) =>
        node.related.some((edge) => !edge.label),
      ),
    [result],
  )

  const handleOpenFolder = useCallback(async () => {
    setLoadingFolder(true)
    setError(null)
    try {
      const s = await openFolderSession()
      if (s) setSession(s)
    } finally {
      setLoadingFolder(false)
    }
  }, [openFolderSession])

  const handleStartScan = async () => {
    setError(null)
    try {
      const j = await api.startCharacterScan()
      setJob(j)
      setResult(null)
    } catch (err) {
      setError(String(err))
    }
  }

  const handleCancelScan = async () => {
    if (!job) return
    try {
      await api.cancelCharacterScan(job.id)
    } catch {
      // best effort
    }
  }

  const handleManage = (character: ScannedCharacter) => {
    setManaging(character)
    setDialogOpen(true)
  }

  const handleSaveCharacter = (updated: ScannedCharacter) => {
    setResult((prev) =>
      prev
        ? {
            ...prev,
            characters: prev.characters.map((c) =>
              c.id === updated.id ? updated : c,
            ),
          }
        : prev,
    )
  }

  const handleSync = async (character: ScannedCharacter) => {
    setSyncingId(character.id)
    setError(null)
    try {
      await api.syncCharacterScanToLibrary([character.id])
      setSyncedIds((prev) => new Set(prev).add(character.id))
    } catch (err) {
      setError(String(err))
    } finally {
      setSyncingId(null)
    }
  }

  const handleSyncAll = async () => {
    if (!result) return
    setSyncingAll(true)
    setError(null)
    try {
      const { syncedIds: synced } = await api.syncCharacterScanToLibrary(
        result.characters.map((c) => c.id),
      )
      setSyncedIds((prev) => new Set([...prev, ...synced]))
    } catch (err) {
      setError(String(err))
    } finally {
      setSyncingAll(false)
    }
  }

  const handleGenerateRelationships = async () => {
    setGeneratingRelationships(true)
    setError(null)
    try {
      const updated = await api.generateCharacterScanRelationships()
      setResult(updated)
    } catch (err) {
      setError(String(err))
    } finally {
      setGeneratingRelationships(false)
    }
  }

  const handleExport = async () => {
    if (!result) return
    setExporting(true)
    setError(null)
    try {
      await api.exportCharacterScan({ ...result, isVerifiedByHuman: true })
      setResult((prev) => (prev ? { ...prev, isVerifiedByHuman: true } : prev))
    } catch (err) {
      setError(String(err))
    } finally {
      setExporting(false)
    }
  }

  return (
    <div className='bg-muted flex h-full min-h-0 flex-1 flex-col overflow-hidden'>
      {/* Header */}
      <div className='border-border bg-background flex h-12 shrink-0 items-center gap-3 border-b px-4'>
        <Link
          href='/'
          prefetch={false}
          className='text-muted-foreground hover:bg-accent hover:text-foreground flex size-8 items-center justify-center rounded-full transition'
        >
          <ChevronLeftIcon className='size-5' />
        </Link>
        <h1 className='text-foreground text-sm font-bold'>
          {t('characterScanner.title')}
        </h1>

        {session && (
          <>
            <Tooltip>
              <TooltipTrigger asChild>
                <span className='text-muted-foreground max-w-xs truncate text-xs'>
                  {session.root}
                </span>
              </TooltipTrigger>
              <TooltipContent side='bottom'>{session.root}</TooltipContent>
            </Tooltip>
            <Button
              size='icon'
              variant='ghost'
              className='size-7 shrink-0'
              onClick={() => void handleOpenFolder()}
              disabled={loadingFolder}
              title={t('characterScanner.pickFolder')}
              aria-label={t('characterScanner.pickFolder')}
            >
              <FolderOpenIcon className='size-4' />
            </Button>
            <div className='flex-1' />
            {isRunning ? (
              <Button size='sm' variant='outline' onClick={handleCancelScan}>
                <StopCircleIcon className='mr-1.5 size-4' />
                {t('characterScanner.cancelScan')}
              </Button>
            ) : (
              <Button size='sm' onClick={() => void handleStartScan()}>
                <PlayIcon className='mr-1.5 size-4' />
                {result
                  ? t('characterScanner.rescan')
                  : t('characterScanner.startScan')}
              </Button>
            )}
            {result && result.characters.length > 0 && (
              <Button
                size='sm'
                variant='outline'
                onClick={() => void handleGenerateRelationships()}
                disabled={generatingRelationships || isRunning || !hasUnlabeledEdges}
                title={
                  !hasUnlabeledEdges
                    ? t('characterScanner.generateRelationships.allLabeled')
                    : undefined
                }
              >
                <SparklesIcon className='mr-1.5 size-4' />
                {generatingRelationships
                  ? t('characterScanner.generateRelationships.generating')
                  : t('characterScanner.generateRelationships.button')}
              </Button>
            )}
            {result && result.characters.length > 0 && (
              <Button
                size='sm'
                variant='outline'
                onClick={() => void handleSyncAll()}
                disabled={syncingAll || syncingId !== null || isRunning}
              >
                <Share2Icon className='mr-1.5 size-4' />
                {syncingAll
                  ? t('characterScanner.syncToLibrary.syncingAll')
                  : t('characterScanner.syncToLibrary.all')}
              </Button>
            )}
            {result && (
              <Button
                size='sm'
                variant='outline'
                onClick={() => void handleExport()}
                disabled={exporting || isRunning}
              >
                <DownloadIcon className='mr-1.5 size-4' />
                {exporting
                  ? t('characterScanner.export.exporting')
                  : t('characterScanner.export.button')}
              </Button>
            )}
          </>
        )}
      </div>

      {/* Progress bar */}
      {isRunning && (
        <div className='border-border bg-background border-b px-4 py-2'>
          <div className='mb-1 flex items-center justify-between text-xs'>
            <span className='text-muted-foreground'>
              {t('characterScanner.progress', {
                current: job.currentDocument,
                total: job.totalDocuments,
              })}
            </span>
            <span className='font-medium'>{job.overallPercent}%</span>
          </div>
          <div className='bg-muted h-1.5 w-full overflow-hidden rounded-full'>
            <div
              className='bg-primary h-full transition-all'
              style={{ width: `${job.overallPercent}%` }}
            />
          </div>
        </div>
      )}

      {error && (
        <div className='border-border border-b bg-red-500/10 px-4 py-2 text-sm text-red-500'>
          {error}
        </div>
      )}

      {/* Body */}
      <div className='min-h-0 flex-1 overflow-y-auto px-4 py-6'>
        <div className='mx-auto max-w-xl'>
          {!session ? (
            <div className='flex flex-col items-center justify-center gap-3 py-16 text-center'>
              <FolderOpenIcon className='text-muted-foreground size-14 opacity-30' />
              <p className='text-muted-foreground text-sm'>
                {t('characterScanner.emptyHint')}
              </p>
              <Button
                variant='outline'
                onClick={() => void handleOpenFolder()}
                disabled={loadingFolder}
              >
                <FolderOpenIcon className='mr-1.5 size-4' />
                {t('characterScanner.pickFolder')}
              </Button>
            </div>
          ) : loadingResult ? (
            <p className='text-muted-foreground py-16 text-center text-sm'>
              {t('characterScanner.loading')}
            </p>
          ) : !result || result.characters.length === 0 ? (
            <div className='flex flex-col items-center justify-center gap-3 py-16 text-center'>
              <ImageIcon className='text-muted-foreground size-14 opacity-30' />
              <p className='text-foreground text-sm font-medium'>
                {t('characterScanner.noResult')}
              </p>
              <p className='text-muted-foreground max-w-sm text-sm'>
                {t('characterScanner.noResultHint')}
              </p>
            </div>
          ) : (
            <Accordion type='multiple' className='border-border border-t'>
              {result.characters.map((character) => (
                <CharacterNode
                  key={character.id}
                  character={character}
                  charactersById={charactersById}
                  relatedEntries={relatedByCharacter.get(character.id) ?? []}
                  onManage={handleManage}
                  onSync={(c) => void handleSync(c)}
                  syncing={syncingId === character.id}
                  synced={syncedIds.has(character.id)}
                />
              ))}
            </Accordion>
          )}
        </div>
      </div>

      <FaceManagerDialog
        character={managing}
        open={dialogOpen}
        onOpenChange={setDialogOpen}
        onSave={handleSaveCharacter}
      />
    </div>
  )
}
