'use client'

import { type ReactNode, useCallback, useEffect, useState } from 'react'
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
  UserIcon,
  Trash2Icon,
} from 'lucide-react'
import { useTranslation } from 'react-i18next'
import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from '@/components/ui/accordion'
import { Button } from '@/components/ui/button'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select'
import { Input } from '@/components/ui/input'
import { Textarea } from '@/components/ui/textarea'
import { subscribeJobChanged } from '@/lib/backend'
import { useDocumentMutations } from '@/lib/query/mutations'
import {
  api,
  type CharacterProfile,
  type CharacterRelation,
  type FolderSessionInfo,
  type StyleProfile,
  type StyleReader,
  type StyleScanResult,
} from '@/lib/api'
import type { JobState } from '@/lib/protocol'

const JOB_KIND = 'style-scan-folder'

type ListKey = 'approach' | 'voice' | 'address' | 'soundEffects'
const LIST_SECTIONS: ListKey[] = [
  'address',
  'approach',
  'voice',
  'soundEffects',
]

const READERS: StyleReader[] = ['vision', 'vietocr']
const VISION_MODELS = ['claude-opus-5', 'claude-sonnet-5', 'claude-haiku-4-5']

const emptyProfile: StyleProfile = {
  approach: [],
  voice: [],
  address: [],
  soundEffects: [],
  glossary: [],
  characters: [],
}

const GENDERS = ['male', 'female'] as const
const AGE_GROUPS = [
  'child',
  'teen',
  'young_adult',
  'adult',
  'middle_age',
  'elder',
] as const
/** Radix Select cannot hold an empty value; this stands for "not known". */
const UNKNOWN = 'unknown'

const splitList = (text: string) =>
  text
    .split(/[,\n]/)
    .map((part) => part.trim())
    .filter(Boolean)

/** An ASCII id from a name, as the backend makes them. */
const slug = (name: string) =>
  name
    .replace(/[đĐ]/g, 'd')
    .normalize('NFD')
    .replace(/[\u0300-\u036f]/g, '')
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-+|-+$/g, '') || 'character'

const newCharacter = (taken: Set<string>): CharacterProfile => {
  let id = 'new-character'
  for (let n = 2; taken.has(id); n++) id = `new-character-${n}`
  return {
    id,
    name: '',
    nameJa: '',
    aliases: [],
    gender: '',
    ageGroup: '',
    role: '',
    personality: '',
    speech: '',
    selfTerms: [],
    appearances: 0,
    relations: [],
  }
}

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

// ─── Characters ─────────────────────────────────────────────────────────────

function Field({ label, children }: { label: string; children: ReactNode }) {
  return (
    <label className='block space-y-1'>
      <span className='text-muted-foreground text-xs'>{label}</span>
      {children}
    </label>
  )
}

/** A text box for a list, one entry per line, committed on blur. */
function ListInput({
  value,
  onChange,
  placeholder,
}: {
  value: string[]
  onChange: (items: string[]) => void
  placeholder?: string
}) {
  const [text, setText] = useState(value.join('\n'))
  useEffect(() => setText(value.join('\n')), [value])
  return (
    <Textarea
      value={text}
      placeholder={placeholder}
      onChange={(e) => setText(e.target.value)}
      onBlur={() => onChange(splitList(text))}
      className='bg-background min-h-9 text-xs'
    />
  )
}

function CharacterCard({
  character,
  cast,
  onChange,
  onRemove,
}: {
  character: CharacterProfile
  cast: CharacterProfile[]
  onChange: (updated: CharacterProfile) => void
  onRemove: () => void
}) {
  const { t } = useTranslation()
  const set = (patch: Partial<CharacterProfile>) =>
    onChange({ ...character, ...patch })
  const setRelation = (index: number, patch: Partial<CharacterRelation>) =>
    set({
      relations: character.relations.map((r, i) =>
        i === index ? { ...r, ...patch } : r,
      ),
    })
  const others = cast.filter((c) => c.id !== character.id)
  const nameOf = (id: string) => cast.find((c) => c.id === id)?.name || id
  const facts = [
    character.gender ? t(`characterScanner.genders.${character.gender}`) : null,
    character.ageGroup
      ? t(`characterScanner.ageGroups.${character.ageGroup}`)
      : null,
    character.appearances
      ? t('styleScanner.characters.pages', { count: character.appearances })
      : null,
  ].filter(Boolean)

  return (
    <AccordionItem value={character.id} className='border-border border-b'>
      <AccordionTrigger className='py-3 hover:no-underline'>
        <div className='flex min-w-0 items-center gap-3 text-left'>
          <div className='bg-muted flex size-8 shrink-0 items-center justify-center rounded-full'>
            <UserIcon className='text-muted-foreground size-4' />
          </div>
          <div className='min-w-0'>
            <div className='text-foreground truncate text-sm font-medium'>
              {character.name || t('styleScanner.characters.unnamed')}
              {character.nameJa && (
                <span className='text-muted-foreground ml-2 font-normal'>
                  {character.nameJa}
                </span>
              )}
            </div>
            <div className='text-muted-foreground truncate text-xs'>
              {[...facts, character.role].filter(Boolean).join(' · ')}
            </div>
          </div>
        </div>
      </AccordionTrigger>
      <AccordionContent>
        <div className='space-y-3 pb-4 pl-11'>
          <div className='grid grid-cols-2 gap-3'>
            <Field label={t('styleScanner.characters.name')}>
              <Input
                value={character.name}
                onChange={(e) => set({ name: e.target.value })}
                className='bg-background h-8 text-xs'
              />
            </Field>
            <Field label={t('styleScanner.characters.nameJa')}>
              <Input
                value={character.nameJa}
                onChange={(e) => set({ nameJa: e.target.value })}
                className='bg-background h-8 text-xs'
              />
            </Field>
            <Field label={t('characterScanner.faceManager.gender')}>
              <Select
                value={character.gender || UNKNOWN}
                onValueChange={(v) => set({ gender: v === UNKNOWN ? '' : v })}
              >
                <SelectTrigger className='bg-background w-full'>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value={UNKNOWN}>
                    {t('characterScanner.faceManager.unknown')}
                  </SelectItem>
                  {GENDERS.map((g) => (
                    <SelectItem key={g} value={g}>
                      {t(`characterScanner.genders.${g}`)}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </Field>
            <Field label={t('characterScanner.faceManager.ageGroup')}>
              <Select
                value={character.ageGroup || UNKNOWN}
                onValueChange={(v) => set({ ageGroup: v === UNKNOWN ? '' : v })}
              >
                <SelectTrigger className='bg-background w-full'>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value={UNKNOWN}>
                    {t('characterScanner.faceManager.unknown')}
                  </SelectItem>
                  {AGE_GROUPS.map((a) => (
                    <SelectItem key={a} value={a}>
                      {t(`characterScanner.ageGroups.${a}`)}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </Field>
          </div>
          <Field label={t('styleScanner.characters.aliases')}>
            <ListInput
              value={character.aliases}
              onChange={(aliases) => set({ aliases })}
            />
          </Field>
          <Field label={t('styleScanner.characters.role')}>
            <Input
              value={character.role}
              onChange={(e) => set({ role: e.target.value })}
              className='bg-background h-8 text-xs'
            />
          </Field>
          <Field label={t('styleScanner.characters.personality')}>
            <Textarea
              value={character.personality}
              onChange={(e) => set({ personality: e.target.value })}
              className='bg-background min-h-9 text-xs'
            />
          </Field>
          <Field label={t('styleScanner.characters.speech')}>
            <Textarea
              value={character.speech}
              onChange={(e) => set({ speech: e.target.value })}
              className='bg-background min-h-9 text-xs'
            />
          </Field>
          <Field label={t('styleScanner.characters.selfTerms')}>
            <ListInput
              value={character.selfTerms}
              onChange={(selfTerms) => set({ selfTerms })}
            />
          </Field>

          <div className='space-y-2'>
            <div className='text-muted-foreground text-xs'>
              {t('styleScanner.characters.relations')}
            </div>
            {character.relations.map((relation, index) => (
              <div
                key={index}
                className='border-border bg-background space-y-2 rounded-md border p-2'
              >
                <div className='flex items-center gap-2'>
                  <span className='text-muted-foreground shrink-0 text-xs'>
                    →
                  </span>
                  <Select
                    value={relation.to || UNKNOWN}
                    onValueChange={(v) =>
                      setRelation(index, { to: v === UNKNOWN ? '' : v })
                    }
                  >
                    <SelectTrigger className='w-44'>
                      <SelectValue>
                        {relation.to
                          ? nameOf(relation.to)
                          : t('styleScanner.characters.pickCharacter')}
                      </SelectValue>
                    </SelectTrigger>
                    <SelectContent>
                      {others.map((o) => (
                        <SelectItem key={o.id} value={o.id}>
                          {o.name || o.id}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                  <Input
                    value={relation.relation}
                    placeholder={t('styleScanner.characters.relationKind')}
                    onChange={(e) =>
                      setRelation(index, { relation: e.target.value })
                    }
                    className='h-7 flex-1 text-xs'
                  />
                  <Button
                    size='icon'
                    variant='ghost'
                    className='size-7 shrink-0'
                    title={t('styleScanner.characters.removeRelation')}
                    onClick={() =>
                      set({
                        relations: character.relations.filter(
                          (_, i) => i !== index,
                        ),
                      })
                    }
                  >
                    <XIcon className='size-4' />
                  </Button>
                </div>
                <Textarea
                  value={relation.address}
                  placeholder={t('styleScanner.characters.relationAddress')}
                  onChange={(e) =>
                    setRelation(index, { address: e.target.value })
                  }
                  className='min-h-9 text-xs'
                />
              </div>
            ))}
            <div className='flex items-center justify-between'>
              <Button
                size='sm'
                variant='outline'
                disabled={others.length === 0}
                onClick={() =>
                  set({
                    relations: [
                      ...character.relations,
                      { to: '', relation: '', address: '' },
                    ],
                  })
                }
              >
                <PlusIcon className='mr-1.5 size-4' />
                {t('styleScanner.characters.addRelation')}
              </Button>
              <Button size='sm' variant='ghost' onClick={onRemove}>
                <Trash2Icon className='mr-1.5 size-4' />
                {t('styleScanner.characters.remove')}
              </Button>
            </div>
          </div>
        </div>
      </AccordionContent>
    </AccordionItem>
  )
}

function CharactersSection({
  characters,
  onChange,
}: {
  characters: CharacterProfile[]
  onChange: (characters: CharacterProfile[]) => void
}) {
  const { t } = useTranslation()
  const [open, setOpen] = useState<string[]>([])

  const update = (index: number, updated: CharacterProfile) =>
    onChange(characters.map((c, i) => (i === index ? updated : c)))

  const remove = (index: number) => {
    const gone = characters[index].id
    onChange(
      characters
        .filter((_, i) => i !== index)
        .map((c) => ({
          ...c,
          relations: c.relations.filter((r) => r.to !== gone),
        })),
    )
  }

  const add = () => {
    const created = newCharacter(new Set(characters.map((c) => c.id)))
    onChange([...characters, created])
    setOpen((prev) => [...prev, created.id])
  }

  return (
    <section className='space-y-2'>
      <div>
        <h2 className='text-foreground text-sm font-semibold'>
          {t('styleScanner.characters.title')}
        </h2>
        <p className='text-muted-foreground text-xs'>
          {t('styleScanner.characters.hint')}
        </p>
      </div>
      {characters.length > 0 && (
        <Accordion
          type='multiple'
          value={open}
          onValueChange={setOpen}
          className='border-border border-t'
        >
          {characters.map((character, index) => (
            <CharacterCard
              key={character.id}
              character={character}
              cast={characters}
              onChange={(updated) => update(index, updated)}
              onRemove={() => remove(index)}
            />
          ))}
        </Accordion>
      )}
      <Button size='sm' variant='outline' onClick={add}>
        <PlusIcon className='mr-1.5 size-4' />
        {t('styleScanner.characters.add')}
      </Button>
    </section>
  )
}

// ─── Page ───────────────────────────────────────────────────────────────────

/** Drop what editing leaves behind: blank lines and half-filled terms. */
function tidy(profile: StyleProfile): StyleProfile {
  const lines = (items: string[] | undefined) =>
    (items ?? []).map((item) => item.trim()).filter(Boolean)
  return {
    approach: lines(profile.approach),
    voice: lines(profile.voice),
    address: lines(profile.address),
    soundEffects: lines(profile.soundEffects),
    // The Japanese side may be empty — a profile read from a Vietnamese
    // edition alone has none.
    glossary: (profile.glossary ?? [])
      .map(([s, t]) => [s.trim(), t.trim()] as [string, string])
      .filter(([, t]) => t),
    characters: tidyCast(profile.characters ?? []),
  }
}

/** Unnamed characters are dropped; a new one takes its id from its name. */
function tidyCast(cast: CharacterProfile[]): CharacterProfile[] {
  const named = cast.filter((c) => c.name.trim())
  const renamed = new Map<string, string>()
  const taken = new Set<string>()
  const withIds = named.map((c) => {
    let id = c.id.startsWith('new-character') ? slug(c.name) : c.id
    for (let n = 2; taken.has(id); n++) id = `${slug(c.name)}-${n}`
    taken.add(id)
    renamed.set(c.id, id)
    return { ...c, id, name: c.name.trim() }
  })
  return withIds.map((c) => ({
    ...c,
    relations: c.relations
      .map((r) => ({ ...r, to: renamed.get(r.to) ?? '' }))
      .filter((r) => r.to && r.to !== c.id),
  }))
}

/** Older files lack sections added since; fill them in so the editor can show them. */
function withAllSections(result: StyleScanResult): StyleScanResult {
  return { ...result, profile: { ...emptyProfile, ...result.profile } }
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
  const [reader, setReader] = useState<StyleReader>('vision')
  const [model, setModel] = useState(VISION_MODELS[0])
  const [library, setLibrary] = useState<StyleScanResult[]>([])

  const loadLibrary = useCallback(async () => {
    try {
      setLibrary(await api.listStyleProfiles())
    } catch {
      // an empty library is not an error
    }
  }, [])

  const loadResult = useCallback(async () => {
    setLoadingResult(true)
    try {
      const loaded = await api.getStyleScanResult()
      setResult(loaded ? withAllSections(loaded) : null)
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
    void loadLibrary()
  }, [loadLibrary])

  useEffect(() => {
    if (session) void loadResult()
  }, [session?.root, loadResult])

  useEffect(() => {
    return subscribeJobChanged((j) => {
      if (j.kind !== JOB_KIND) return
      setJob(j)
      if (j.status === 'completed') {
        void loadResult()
        void loadLibrary()
      }
      if (j.status === 'failed' && j.error) setError(j.error)
    })
  }, [loadResult, loadLibrary])

  const isRunning = job?.status === 'running'
  const isActive = !!result && sameProfile(tidy(result.profile), active)
  const isInUse = (profile: StyleProfile) =>
    !!active && sameProfile(tidy(profile), tidy(active))

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
      setJob(
        await api.startStyleScan(
          reader === 'vision' ? { reader, model } : { reader },
        ),
      )
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
    void loadLibrary()
    return saved
  }

  const handleUseSaved = async (entry: StyleScanResult) => {
    setSwitching(true)
    setError(null)
    try {
      const profile = isInUse(entry.profile) ? null : tidy(entry.profile)
      await api.setActiveStyleProfile(profile)
      setActive(profile)
    } catch (err) {
      setError(String(err))
    } finally {
      setSwitching(false)
    }
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
            {!isRunning && (
              <>
                <Select
                  value={reader}
                  onValueChange={(v) => setReader(v as StyleReader)}
                >
                  <SelectTrigger size='sm' className='w-44'>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    {READERS.map((r) => (
                      <SelectItem key={r} value={r}>
                        {t(`styleScanner.reader.${r}`)}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
                {reader === 'vision' && (
                  <Select value={model} onValueChange={setModel}>
                    <SelectTrigger size='sm' className='w-40'>
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      {VISION_MODELS.map((m) => (
                        <SelectItem key={m} value={m}>
                          {m}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                )}
              </>
            )}
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
              <p className='text-muted-foreground max-w-md text-xs'>
                {t('styleScanner.claudeHint')}{' '}
                <code className='bg-muted rounded px-1 py-0.5'>
                  /style-read {session.root}
                </code>
              </p>
            </div>
          ) : (
            <div className='space-y-6'>
              <p className='text-muted-foreground text-xs'>
                {result.source === 'claude-api'
                  ? t(
                      result.withRaw
                        ? 'styleScanner.statsClaudeApi'
                        : 'styleScanner.statsClaudeApiVietnamese',
                      {
                        pages: result.pairedPages,
                        lines: result.pairCount,
                        model: result.model ?? '',
                      },
                    )
                  : result.source === 'claude'
                    ? result.translatedPages > 0
                      ? t('styleScanner.statsClaude', {
                          pages: result.pairedPages,
                          total: result.translatedPages,
                        })
                      : t('styleScanner.statsClaudeRaw', {
                          pages: result.pairedPages,
                          total: result.rawPages,
                        })
                    : t('styleScanner.stats', {
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
              <CharactersSection
                characters={result.profile.characters ?? []}
                onChange={(characters) => editProfile({ characters })}
              />
              {LIST_SECTIONS.map((section) => (
                <ListSection
                  key={section}
                  section={section}
                  items={result.profile[section] ?? []}
                  onChange={(items) => editProfile({ [section]: items })}
                />
              ))}
              <GlossarySection
                entries={result.profile.glossary}
                onChange={(glossary) => editProfile({ glossary })}
              />
            </div>
          )}

          <section className='border-border mt-10 space-y-2 border-t pt-6'>
            <div>
              <h2 className='text-foreground text-sm font-semibold'>
                {t('styleScanner.library.title')}
              </h2>
              <p className='text-muted-foreground text-xs'>
                {t('styleScanner.library.hint')}
              </p>
            </div>
            {library.length === 0 ? (
              <p className='text-muted-foreground text-xs'>
                {t('styleScanner.library.empty')}
              </p>
            ) : (
              <ul className='divide-border divide-y'>
                {library.map((entry) => {
                  const inUse = isInUse(entry.profile)
                  return (
                    <li
                      key={entry.name}
                      className='flex items-center gap-3 py-2 text-sm'
                    >
                      <div className='min-w-0 flex-1'>
                        <div className='text-foreground truncate font-medium'>
                          {entry.name}
                        </div>
                        <div className='text-muted-foreground truncate text-xs'>
                          {[
                            entry.model ?? entry.source,
                            t('styleScanner.library.pages', {
                              count: entry.pairedPages,
                            }),
                            entry.withRaw === false
                              ? t('styleScanner.library.vietnameseOnly')
                              : null,
                            entry.isVerifiedByHuman
                              ? t('styleScanner.reviewed')
                              : null,
                          ]
                            .filter(Boolean)
                            .join(' · ')}
                        </div>
                      </div>
                      <Button
                        size='sm'
                        variant={inUse ? 'secondary' : 'outline'}
                        disabled={switching}
                        onClick={() => void handleUseSaved(entry)}
                        title={t('styleScanner.activeHint')}
                      >
                        <CheckIcon className='mr-1.5 size-4' />
                        {inUse
                          ? t('styleScanner.inUse')
                          : t('styleScanner.use')}
                      </Button>
                    </li>
                  )
                })}
              </ul>
            )}
          </section>
        </div>
      </div>
    </div>
  )
}
