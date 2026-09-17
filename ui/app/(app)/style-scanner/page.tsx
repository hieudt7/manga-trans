'use client'

import { useCallback, useEffect, useState } from 'react'
import Link from 'next/link'
import {
  ChevronLeftIcon,
  FolderOpenIcon,
  PlayIcon,
  StopCircleIcon,
  ImageIcon,
  SaveIcon,
  CheckIcon,
  PlusIcon,
  XIcon,
} from 'lucide-react'
import { useTranslation } from 'react-i18next'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Textarea } from '@/components/ui/textarea'
import { subscribeJobChanged } from '@/lib/backend'
import { useDocumentMutations } from '@/lib/query/mutations'
import {
  api,
  type FolderSessionInfo,
  type StyleProfile,
  type StyleScanResult,
} from '@/lib/api'
import type { JobState } from '@/lib/protocol'

const JOB_KIND = 'style-scan-folder'

type ListKey = 'voice' | 'address' | 'soundEffects'
const LIST_SECTIONS: ListKey[] = ['address', 'voice', 'soundEffects']

const sameProfile = (a: StyleProfile | null, b: StyleProfile | null) =>
  JSON.stringify(a) === JSON.stringify(b)

// ─── Editable list (one observation per line) ───────────────────────────────

function ListSection({
  section,
  items,
  onChange,
}: {
  section: ListKey
  items: string[]
  onChange: (items: string[]) => void
}) {
  const { t } = useTranslation()
  // Edited as text so a line can be split, joined or reordered freely; blank
  // lines are dropped only when the profile is saved.
  const [text, setText] = useState(items.join('\n'))
  useEffect(() => setText(items.join('\n')), [items])

  return (
    <section className='space-y-2'>
      <div>
        <h2 className='text-foreground text-sm font-semibold'>
          {t(`styleScanner.sections.${section}.title`)}
        </h2>
        <p className='text-muted-foreground text-xs'>
          {t(`styleScanner.sections.${section}.hint`)}
        </p>
      </div>
      <Textarea
        value={text}
        onChange={(e) => setText(e.target.value)}
        onBlur={() => onChange(text.split('\n'))}
        placeholder={t('styleScanner.emptySection')}
        className='bg-background font-mono text-xs leading-relaxed'
      />
    </section>
  )
}

// ─── Glossary ───────────────────────────────────────────────────────────────

function GlossarySection({
  entries,
  onChange,
}: {
  entries: [string, string][]
  onChange: (entries: [string, string][]) => void
}) {
  const { t } = useTranslation()

  const update = (index: number, side: 0 | 1, value: string) =>
    onChange(
      entries.map((entry, i) => {
        if (i !== index) return entry
        const next: [string, string] = [entry[0], entry[1]]
        next[side] = value
        return next
      }),
    )

  return (
    <section className='space-y-2'>
      <div>
        <h2 className='text-foreground text-sm font-semibold'>
          {t('styleScanner.sections.glossary.title')}
        </h2>
        <p className='text-muted-foreground text-xs'>
          {t('styleScanner.sections.glossary.hint')}
        </p>
      </div>
      <div className='space-y-1.5'>
        {entries.map(([source, target], index) => (
          <div key={index} className='flex items-center gap-2'>
            <Input
              value={source}
              onChange={(e) => update(index, 0, e.target.value)}
              placeholder={t('styleScanner.glossary.source')}
              className='bg-background h-8 flex-1 text-xs'
            />
            <span className='text-muted-foreground text-xs'>→</span>
            <Input
              value={target}
              onChange={(e) => update(index, 1, e.target.value)}
              placeholder={t('styleScanner.glossary.target')}
              className='bg-background h-8 flex-1 text-xs'
            />
            <Button
              size='icon'
              variant='ghost'
              className='size-8 shrink-0'
              onClick={() => onChange(entries.filter((_, i) => i !== index))}
              title={t('styleScanner.glossary.remove')}
            >
              <XIcon className='size-4' />
            </Button>
          </div>
        ))}
        <Button
          size='sm'
          variant='outline'
          onClick={() => onChange([...entries, ['', '']])}
        >
          <PlusIcon className='mr-1.5 size-4' />
          {t('styleScanner.glossary.add')}
        </Button>
      </div>
    </section>
  )
}

// ─── Page ───────────────────────────────────────────────────────────────────

/** Drop what editing leaves behind: blank lines and half-filled terms. */
function tidy(profile: StyleProfile): StyleProfile {
  const lines = (items: string[]) =>
    items.map((item) => item.trim()).filter(Boolean)
  return {
    voice: lines(profile.voice),
    address: lines(profile.address),
    soundEffects: lines(profile.soundEffects),
    glossary: profile.glossary
      .map(([s, t]) => [s.trim(), t.trim()] as [string, string])
      .filter(([s, t]) => s && t),
  }
}

export default function StyleScannerPage() {
  const { t } = useTranslation()
  const { openFolderSession } = useDocumentMutations()

  const [session, setSession] = useState<FolderSessionInfo | null>(null)
  const [loadingFolder, setLoadingFolder] = useState(false)
  const [job, setJob] = useState<JobState | null>(null)
  const [result, setResult] = useState<StyleScanResult | null>(null)
  const [active, setActive] = useState<StyleProfile | null>(null)
  const [loadingResult, setLoadingResult] = useState(false)
  const [dirty, setDirty] = useState(false)
  const [saving, setSaving] = useState(false)
  const [switching, setSwitching] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const loadResult = useCallback(async () => {
    setLoadingResult(true)
    try {
      setResult(await api.getStyleScanResult())
      setDirty(false)
    } catch {
      // no result yet — not an error the user needs to see
    } finally {
      setLoadingResult(false)
    }
  }, [])

  useEffect(() => {
    void (async () => {
      try {
        const s = await api.getFolderSession()
        if (s) setSession(s)
      } catch {
        // no active session
      }
      try {
        setActive(await api.getActiveStyleProfile())
      } catch {
        // nothing active
      }
    })()
  }, [])

  useEffect(() => {
    if (session) void loadResult()
  }, [session?.root, loadResult])

  useEffect(() => {
    return subscribeJobChanged((j) => {
      if (j.kind !== JOB_KIND) return
      setJob(j)
      if (j.status === 'completed') void loadResult()
      if (j.status === 'failed' && j.error) setError(j.error)
    })
  }, [loadResult])

  const isRunning = job?.status === 'running'
  const isActive = !!result && sameProfile(tidy(result.profile), active)

  const editProfile = (patch: Partial<StyleProfile>) => {
    setResult((prev) =>
      prev ? { ...prev, profile: { ...prev.profile, ...patch } } : prev,
    )
    setDirty(true)
  }

  const handleOpenFolder = async () => {
    setLoadingFolder(true)
    setError(null)
    try {
      const s = await openFolderSession()
      if (s) setSession(s)
    } finally {
      setLoadingFolder(false)
    }
  }

  const handleStartScan = async () => {
    setError(null)
    try {
      setJob(await api.startStyleScan())
    } catch (err) {
      setError(String(err))
    }
  }

  const handleCancelScan = async () => {
    if (!job) return
    try {
      await api.cancelStyleScan(job.id)
    } catch {
      // best effort
    }
  }

  const save = async (): Promise<StyleScanResult | null> => {
    if (!result) return null
    const saved = {
      ...result,
      profile: tidy(result.profile),
      isVerifiedByHuman: true,
    }
    await api.exportStyleScan(saved)
    setResult(saved)
    setDirty(false)
    return saved
  }

  const handleSave = async () => {
    setSaving(true)
    setError(null)
    try {
      const saved = await save()
      // Keep the translator in step with the edit when this is the profile in use.
      if (saved && isActive) {
        await api.setActiveStyleProfile(saved.profile)
        setActive(saved.profile)
      }
    } catch (err) {
      setError(String(err))
    } finally {
      setSaving(false)
    }
  }

  const handleToggleActive = async () => {
    setSwitching(true)
    setError(null)
    try {
      if (isActive) {
        await api.setActiveStyleProfile(null)
        setActive(null)
      } else {
        const saved = await save()
        if (saved) {
          await api.setActiveStyleProfile(saved.profile)
          setActive(saved.profile)
        }
      }
    } catch (err) {
      setError(String(err))
    } finally {
      setSwitching(false)
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
          {t('styleScanner.title')}
        </h1>

        {session && (
          <>
            <span className='text-muted-foreground max-w-xs truncate text-xs'>
              {session.root}
            </span>
            <div className='flex-1' />
            {isRunning ? (
              <Button size='sm' variant='outline' onClick={handleCancelScan}>
                <StopCircleIcon className='mr-1.5 size-4' />
                {t('styleScanner.cancelScan')}
              </Button>
            ) : (
              <Button size='sm' onClick={() => void handleStartScan()}>
                <PlayIcon className='mr-1.5 size-4' />
                {result
                  ? t('styleScanner.rescan')
                  : t('styleScanner.startScan')}
              </Button>
            )}
            {result && (
              <>
                <Button
                  size='sm'
                  variant='outline'
                  onClick={() => void handleSave()}
                  disabled={saving || isRunning || !dirty}
                >
                  <SaveIcon className='mr-1.5 size-4' />
                  {saving ? t('styleScanner.saving') : t('styleScanner.save')}
                </Button>
                <Button
                  size='sm'
                  variant={isActive ? 'secondary' : 'outline'}
                  onClick={() => void handleToggleActive()}
                  disabled={switching || isRunning}
                  title={t('styleScanner.activeHint')}
                >
                  <CheckIcon className='mr-1.5 size-4' />
                  {isActive ? t('styleScanner.inUse') : t('styleScanner.use')}
                </Button>
              </>
            )}
          </>
        )}
      </div>

      {/* Progress bar */}
      {isRunning && (
        <div className='border-border bg-background border-b px-4 py-2'>
          <div className='mb-1 flex items-center justify-between text-xs'>
            <span className='text-muted-foreground'>
              {t('styleScanner.progress', {
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
        <div className='mx-auto max-w-2xl'>
          {!session ? (
            <div className='flex flex-col items-center justify-center gap-3 py-16 text-center'>
              <FolderOpenIcon className='text-muted-foreground size-14 opacity-30' />
              <p className='text-muted-foreground max-w-md text-sm'>
                {t('styleScanner.emptyHint')}
              </p>
              <Button
                variant='outline'
                onClick={() => void handleOpenFolder()}
                disabled={loadingFolder}
              >
                <FolderOpenIcon className='mr-1.5 size-4' />
                {t('styleScanner.pickFolder')}
              </Button>
            </div>
          ) : loadingResult ? (
            <p className='text-muted-foreground py-16 text-center text-sm'>
              {t('styleScanner.loading')}
            </p>
          ) : !result ? (
            <div className='flex flex-col items-center justify-center gap-3 py-16 text-center'>
              <ImageIcon className='text-muted-foreground size-14 opacity-30' />
              <p className='text-foreground text-sm font-medium'>
                {t('styleScanner.noResult')}
              </p>
              <p className='text-muted-foreground max-w-md text-sm'>
                {t('styleScanner.noResultHint')}
              </p>
            </div>
          ) : (
            <div className='space-y-6'>
              <p className='text-muted-foreground text-xs'>
                {t('styleScanner.stats', {
                  paired: result.pairedPages,
                  raw: result.rawPages,
                  translated: result.translatedPages,
                  pairs: result.pairCount,
                })}
                {' · '}
                {result.isVerifiedByHuman
                  ? t('styleScanner.reviewed')
                  : t('styleScanner.notReviewed')}
              </p>
              {LIST_SECTIONS.map((section) => (
                <ListSection
                  key={section}
                  section={section}
                  items={result.profile[section]}
                  onChange={(items) => editProfile({ [section]: items })}
                />
              ))}
              <GlossarySection
                entries={result.profile.glossary}
                onChange={(glossary) => editProfile({ glossary })}
              />
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
