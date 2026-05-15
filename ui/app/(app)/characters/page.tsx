'use client'

import { useCallback, useEffect, useRef, useState, type DragEvent } from 'react'
import Link from 'next/link'
import {
  ChevronLeftIcon,
  PlusIcon,
  XIcon,
  Trash2Icon,
  UserIcon,
  ImageIcon,
} from 'lucide-react'
import { ScrollArea } from '@/components/ui/scroll-area'
import { useTranslation } from 'react-i18next'
import { api, type CharacterInfo } from '@/lib/api'

const inputClass =
  'border-border bg-card text-foreground placeholder:text-muted-foreground focus:ring-primary w-full rounded-md border px-3 py-1.5 text-sm focus:ring-1 focus:outline-none'

// ─── Face drop zone (multi-image) ─────────────────────────────────────────────

function FaceDropZone({
  files,
  onChange,
}: {
  files: File[]
  onChange: (files: File[]) => void
}) {
  const { t } = useTranslation()
  const inputRef = useRef<HTMLInputElement>(null)
  const [dragging, setDragging] = useState(false)

  const addFiles = (incoming: FileList | null) => {
    if (!incoming) return
    const valid = Array.from(incoming).filter((f) =>
      f.type.startsWith('image/'),
    )
    if (valid.length) onChange([...files, ...valid])
  }

  const remove = (idx: number) => onChange(files.filter((_, i) => i !== idx))

  const onDrop = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault()
    setDragging(false)
    addFiles(e.dataTransfer.files)
  }

  return (
    <div className='space-y-2'>
      {/* Thumbnails row */}
      {files.length > 0 && (
        <div className='flex flex-wrap gap-2'>
          {files.map((f, i) => {
            const url = URL.createObjectURL(f)
            return (
              <div key={i} className='relative size-16 shrink-0'>
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img
                  src={url}
                  alt={`face ${i + 1}`}
                  className='border-border h-full w-full rounded-md border object-cover'
                />
                <button
                  type='button'
                  onClick={() => remove(i)}
                  className='bg-background/80 hover:bg-background absolute -top-1 -right-1 rounded-full p-0.5'
                >
                  <XIcon className='size-3' />
                </button>
              </div>
            )
          })}
        </div>
      )}

      {/* Drop target */}
      <div
        onClick={() => inputRef.current?.click()}
        onDragOver={(e) => {
          e.preventDefault()
          setDragging(true)
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={onDrop}
        data-dragging={dragging}
        className='border-border bg-card hover:border-foreground/30 data-[dragging=true]:border-primary flex h-20 cursor-pointer items-center justify-center rounded-lg border-2 border-dashed transition select-none'
      >
        <input
          ref={inputRef}
          type='file'
          accept='image/*'
          multiple
          className='hidden'
          onChange={(e) => {
            addFiles(e.target.files)
            e.target.value = ''
          }}
        />
        <div className='text-muted-foreground flex flex-col items-center gap-1'>
          <ImageIcon className='size-5' />
          <span className='text-xs'>{t('characters.faceDrop')}</span>
        </div>
      </div>
    </div>
  )
}

// ─── Trait chip input ──────────────────────────────────────────────────────────

function TraitInput({
  traits,
  onChange,
}: {
  traits: string[]
  onChange: (traits: string[]) => void
}) {
  const { t } = useTranslation()
  const [draft, setDraft] = useState('')

  const commit = () => {
    const v = draft.trim()
    if (v && !traits.includes(v)) onChange([...traits, v])
    setDraft('')
  }

  return (
    <div className='space-y-2'>
      <div className='flex flex-wrap gap-1.5'>
        {traits.map((trait) => (
          <span
            key={trait}
            className='bg-accent text-foreground inline-flex items-center gap-1 rounded-full px-2.5 py-0.5 text-xs font-medium'
          >
            {trait}
            <button
              type='button'
              onClick={() => onChange(traits.filter((t) => t !== trait))}
              className='text-muted-foreground hover:text-foreground'
            >
              <XIcon className='size-2.5' />
            </button>
          </span>
        ))}
      </div>
      <input
        type='text'
        value={draft}
        onChange={(e) => setDraft(e.target.value)}
        onKeyDown={(e) => {
          if (e.key === 'Enter' || e.key === ',') {
            e.preventDefault()
            commit()
          }
        }}
        onBlur={commit}
        placeholder={t('characters.traitsPlaceholder')}
        className={inputClass}
      />
    </div>
  )
}

// ─── Relation list input ───────────────────────────────────────────────────────

function RelationList({
  relations,
  onChange,
}: {
  relations: string[]
  onChange: (r: string[]) => void
}) {
  const { t } = useTranslation()
  const [draft, setDraft] = useState('')

  const add = () => {
    const v = draft.trim()
    if (v) {
      onChange([...relations, v])
      setDraft('')
    }
  }

  return (
    <div className='space-y-2'>
      {relations.map((rel, i) => (
        <div key={i} className='flex items-center gap-2'>
          <span className='border-border bg-card flex-1 rounded-md border px-3 py-1.5 text-sm'>
            {rel}
          </span>
          <button
            type='button'
            onClick={() => onChange(relations.filter((_, j) => j !== i))}
            className='text-muted-foreground hover:text-foreground'
          >
            <XIcon className='size-4' />
          </button>
        </div>
      ))}
      <div className='flex gap-2'>
        <input
          type='text'
          value={draft}
          onChange={(e) => setDraft(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === 'Enter') {
              e.preventDefault()
              add()
            }
          }}
          placeholder={t('characters.relationsPlaceholder')}
          className={`${inputClass} flex-1`}
        />
        <button
          type='button'
          onClick={add}
          disabled={!draft.trim()}
          className='border-border bg-card text-foreground hover:bg-accent disabled:text-muted-foreground inline-flex items-center gap-1 rounded-md border px-3 py-1.5 text-sm font-medium transition disabled:opacity-50'
        >
          <PlusIcon className='size-3.5' />
          {t('characters.addRelation')}
        </button>
      </div>
    </div>
  )
}

// ─── Add character form ────────────────────────────────────────────────────────

function AddCharacterForm({ onDone }: { onDone: () => void }) {
  const { t } = useTranslation()
  const [name, setName] = useState('')
  const [faces, setFaces] = useState<File[]>([])
  const [traits, setTraits] = useState<string[]>([])
  const [relations, setRelations] = useState<string[]>([])
  const [saving, setSaving] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!name.trim() || faces.length === 0) return
    setSaving(true)
    setError(null)
    try {
      await api.addCharacter({ name: name.trim(), faces, traits, relations })
      onDone()
    } catch (err) {
      setError(String(err))
      setSaving(false)
    }
  }

  return (
    <form
      onSubmit={handleSubmit}
      className='border-border bg-card mb-6 space-y-4 rounded-xl border p-4'
    >
      <h2 className='text-foreground text-sm font-bold'>
        {t('characters.add')}
      </h2>

      {/* Face image */}
      <div className='space-y-1'>
        <label className='text-foreground text-sm'>
          {t('characters.face')}
        </label>
        <FaceDropZone files={faces} onChange={setFaces} />
        <p className='text-muted-foreground text-xs'>
          {t('characters.faceHint')}
        </p>
      </div>

      {/* Name */}
      <div className='space-y-1'>
        <label className='text-foreground text-sm'>
          {t('characters.name')}
        </label>
        <input
          type='text'
          value={name}
          onChange={(e) => setName(e.target.value)}
          placeholder={t('characters.namePlaceholder')}
          className={inputClass}
          required
        />
      </div>

      {/* Traits */}
      <div className='space-y-1'>
        <label className='text-foreground text-sm'>
          {t('characters.traits')}
        </label>
        <TraitInput traits={traits} onChange={setTraits} />
      </div>

      {/* Relations & address terms */}
      <div className='space-y-1'>
        <label className='text-foreground text-sm'>
          {t('characters.relations')}
        </label>
        <p className='text-muted-foreground text-xs'>
          {t('characters.relationsHint')}
        </p>
        <RelationList relations={relations} onChange={setRelations} />
      </div>

      {error && <p className='text-sm text-red-500'>{error}</p>}

      <div className='flex justify-end gap-2'>
        <button
          type='button'
          onClick={onDone}
          className='border-border bg-card text-foreground hover:bg-accent rounded-md border px-4 py-1.5 text-sm font-medium transition'
        >
          {t('characters.cancel')}
        </button>
        <button
          type='submit'
          disabled={saving || !name.trim() || faces.length === 0}
          className='bg-primary text-primary-foreground hover:bg-primary/90 disabled:bg-primary/50 rounded-md px-4 py-1.5 text-sm font-medium transition disabled:cursor-not-allowed'
        >
          {saving ? t('characters.saving') : t('characters.confirm')}
        </button>
      </div>
    </form>
  )
}

// ─── Character card ────────────────────────────────────────────────────────────

function CharacterCard({
  character,
  onDelete,
}: {
  character: CharacterInfo
  onDelete: () => void
}) {
  return (
    <div className='border-border bg-card flex items-start gap-3 rounded-xl border p-4'>
      <div className='bg-muted flex size-10 shrink-0 items-center justify-center rounded-full'>
        <UserIcon className='text-muted-foreground size-5' />
      </div>
      <div className='min-w-0 flex-1'>
        <p className='text-foreground font-medium'>{character.name}</p>
        {character.traits.length > 0 && (
          <div className='mt-1.5 flex flex-wrap gap-1'>
            {character.traits.map((t) => (
              <span
                key={t}
                className='bg-accent text-foreground rounded-full px-2 py-0.5 text-xs'
              >
                {t}
              </span>
            ))}
          </div>
        )}
        {character.relations.length > 0 && (
          <ul className='text-muted-foreground mt-2 space-y-0.5 text-xs'>
            {character.relations.map((r, i) => (
              <li key={i}>{r}</li>
            ))}
          </ul>
        )}
      </div>
      <button
        type='button'
        onClick={onDelete}
        className='text-muted-foreground hover:text-destructive shrink-0 transition'
        aria-label='Delete'
      >
        <Trash2Icon className='size-4' />
      </button>
    </div>
  )
}

// ─── Page ──────────────────────────────────────────────────────────────────────

export default function CharactersPage() {
  const { t } = useTranslation()
  const [characters, setCharacters] = useState<CharacterInfo[]>([])
  const [showForm, setShowForm] = useState(false)
  const [loading, setLoading] = useState(true)

  const load = useCallback(async () => {
    try {
      const list = await api.listCharacters()
      setCharacters(list)
    } catch {
      // silently fail — character feature may not be initialised
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    void load()
  }, [load])

  const handleDelete = async (id: string) => {
    try {
      await api.removeCharacter(id)
      setCharacters((prev) => prev.filter((c) => c.id !== id))
    } catch (err) {
      console.error('Failed to remove character', err)
    }
  }

  const handleAdded = () => {
    setShowForm(false)
    void load()
  }

  return (
    <div className='bg-muted flex min-h-0 flex-1 flex-col overflow-hidden'>
      <ScrollArea className='min-h-0 flex-1' viewportClassName='h-full'>
        <div className='min-h-full px-4 py-6'>
          <div className='relative mx-auto max-w-xl'>
            {/* Header */}
            <div className='mb-8 flex items-center justify-between'>
              <div className='flex items-center'>
                <Link
                  href='/'
                  prefetch={false}
                  className='text-muted-foreground hover:bg-accent hover:text-foreground absolute -left-14 flex size-10 items-center justify-center rounded-full transition'
                >
                  <ChevronLeftIcon className='size-6' />
                </Link>
                <h1 className='text-foreground text-2xl font-bold'>
                  {t('characters.title')}
                </h1>
              </div>
              {!showForm && (
                <button
                  type='button'
                  onClick={() => setShowForm(true)}
                  className='border-border bg-card text-foreground hover:bg-accent inline-flex items-center gap-1.5 rounded-lg border px-3 py-1.5 text-sm font-medium transition'
                >
                  <PlusIcon className='size-4' />
                  {t('characters.add')}
                </button>
              )}
            </div>

            <p className='text-muted-foreground mb-6 text-sm'>
              {t('characters.description')}
            </p>

            {/* Add form */}
            {showForm && <AddCharacterForm onDone={handleAdded} />}

            {/* Character list */}
            {!loading && characters.length === 0 && !showForm && (
              <div className='py-12 text-center'>
                <UserIcon className='text-muted-foreground mx-auto mb-3 size-10' />
                <p className='text-foreground text-sm font-medium'>
                  {t('characters.empty')}
                </p>
                <p className='text-muted-foreground mt-1 text-sm'>
                  {t('characters.emptyHint')}
                </p>
              </div>
            )}

            <div className='space-y-3'>
              {characters.map((character) => (
                <CharacterCard
                  key={character.id}
                  character={character}
                  onDelete={() => void handleDelete(character.id)}
                />
              ))}
            </div>
          </div>
        </div>
      </ScrollArea>
    </div>
  )
}
