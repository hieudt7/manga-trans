'use client'

import { useEffect, useMemo, useState } from 'react'
import { useTranslation } from 'react-i18next'
import { motion } from 'motion/react'
import {
  ScanIcon,
  ScanTextIcon,
  MessageCircleIcon,
  Wand2Icon,
  TypeIcon,
  LoaderCircleIcon,
  LanguagesIcon,
  ZapIcon,
  UsersIcon,
} from 'lucide-react'
import { Separator } from '@/components/ui/separator'
import { Button } from '@/components/ui/button'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select'
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from '@/components/ui/popover'
import { useLlmUiStore } from '@/lib/stores/llmUiStore'
import { useEditorUiStore } from '@/lib/stores/editorUiStore'
import {
  useLlmModelsQuery,
  useLlmReadyQuery,
  LOCAL_LLM_PRESET_LABELS,
} from '@/lib/query/hooks'
import { useDocumentMutations, useLlmMutations } from '@/lib/query/mutations'
import { useOperationStore } from '@/lib/stores/operationStore'
import { usePreferencesStore } from '@/lib/stores/preferencesStore'
import { getProviderDisplayName } from '@/lib/providers'
import { api } from '@/lib/api'

const TRANSLATION_INCOMPLETE_PREFIX = 'TRANSLATION_INCOMPLETE:'

function parseTranslationIncomplete(
  error: unknown,
): { received: number; expected: number } | null {
  const msg = error instanceof Error ? error.message : String(error)
  const idx = msg.indexOf(TRANSLATION_INCOMPLETE_PREFIX)
  if (idx === -1) return null
  const [received, expected] = msg
    .slice(idx + TRANSLATION_INCOMPLETE_PREFIX.length)
    .split('/')
    .map(Number)
  if (isNaN(received) || isNaN(expected)) return null
  return { received, expected }
}

export function CanvasToolbar() {
  return (
    <div className='border-border/60 bg-card text-foreground flex items-center gap-2 border-b px-3 py-2 text-xs'>
      <div className='flex min-w-0 flex-1 items-center overflow-x-auto'>
        <WorkflowButtons />
      </div>
      <LlmStatusPopover />
    </div>
  )
}

function WorkflowButtons() {
  const { inpaint, detect, ocr, detectBalloon, render, saveRendered } =
    useDocumentMutations()
  const { llmGenerate } = useLlmMutations()
  const { data: llmReady = false } = useLlmReadyQuery()
  const [generating, setGenerating] = useState(false)
  const [processingAll, setProcessingAll] = useState(false)
  const { t } = useTranslation()
  const operation = useOperationStore((state) => state.operation)

  const isDetecting =
    operation?.type === 'process-current' && operation?.step === 'detect'
  const isOcr =
    operation?.type === 'process-current' && operation?.step === 'ocr'
  const isDetectingBalloon =
    operation?.type === 'process-current' &&
    operation?.step === 'detect-balloon'
  const isInpainting =
    operation?.type === 'process-current' && operation?.step === 'inpaint'
  const isRendering =
    operation?.type === 'process-current' && operation?.step === 'render'

  const isBusy =
    processingAll ||
    isDetecting ||
    isOcr ||
    isInpainting ||
    isRendering ||
    generating

  const handleTranslate = async () => {
    setGenerating(true)
    try {
      await llmGenerate(null)
    } catch (error) {
      const incomplete = parseTranslationIncomplete(error)
      if (incomplete) {
        window.alert(
          t('llm.translationIncomplete', {
            received: incomplete.received,
            expected: incomplete.expected,
          }),
        )
      } else {
        console.error(error)
      }
    } finally {
      setGenerating(false)
    }
  }

  const handleProcessAll = async () => {
    const { totalPages, setCurrentDocumentIndex } = useEditorUiStore.getState()
    setProcessingAll(true)
    try {
      for (let i = 0; i < totalPages; i++) {
        setCurrentDocumentIndex(i)
        await detect(null, i).catch(console.error)
        await ocr(null, i).catch(console.error)
        await detectBalloon(null, i).catch(console.error)
        if (llmReady) {
          try {
            await llmGenerate(null, i)
          } catch (error) {
            const incomplete = parseTranslationIncomplete(error)
            if (incomplete) {
              const shouldContinue = window.confirm(
                t('llm.translationIncompleteConfirm', {
                  page: i + 1,
                  received: incomplete.received,
                  expected: incomplete.expected,
                }),
              )
              if (!shouldContinue) break
            } else {
              console.error(error)
            }
          }
        }
        await inpaint(null, i).catch(console.error)
        await render(null, i).catch(console.error)
        await saveRendered(null, i).catch(console.error)
      }
    } finally {
      setProcessingAll(false)
    }
  }

  return (
    <div className='flex items-center gap-0.5'>
      <Button
        variant='ghost'
        size='xs'
        onClick={detect}
        data-testid='toolbar-detect'
        disabled={isBusy}
      >
        {isDetecting ? (
          <LoaderCircleIcon className='size-4 animate-spin' />
        ) : (
          <ScanIcon className='size-4' />
        )}
        {t('processing.detect')}
      </Button>

      <Separator orientation='vertical' className='mx-0.5 h-4' />

      <Button
        variant='ghost'
        size='xs'
        onClick={ocr}
        data-testid='toolbar-ocr'
        disabled={isBusy}
      >
        {isOcr ? (
          <LoaderCircleIcon className='size-4 animate-spin' />
        ) : (
          <ScanTextIcon className='size-4' />
        )}
        {t('processing.ocr')}
      </Button>

      <Separator orientation='vertical' className='mx-0.5 h-4' />

      <Button
        variant='ghost'
        size='xs'
        onClick={detectBalloon}
        data-testid='toolbar-detect-balloon'
        disabled={isBusy}
      >
        {isDetectingBalloon ? (
          <LoaderCircleIcon className='size-4 animate-spin' />
        ) : (
          <MessageCircleIcon className='size-4' />
        )}
        {t('processing.detectBalloon')}
      </Button>

      <Separator orientation='vertical' className='mx-0.5 h-4' />

      <Button
        variant='ghost'
        size='xs'
        onClick={handleTranslate}
        disabled={!llmReady || isBusy}
        data-testid='toolbar-translate'
      >
        {generating ? (
          <LoaderCircleIcon className='size-4 animate-spin' />
        ) : (
          <LanguagesIcon className='size-4' />
        )}
        {t('llm.generate')}
      </Button>

      <Separator orientation='vertical' className='mx-0.5 h-4' />

      <CharacterDebugPopover />

      <Separator orientation='vertical' className='mx-0.5 h-4' />

      <Button
        variant='ghost'
        size='xs'
        onClick={inpaint}
        data-testid='toolbar-inpaint'
        disabled={isBusy}
      >
        {isInpainting ? (
          <LoaderCircleIcon className='size-4 animate-spin' />
        ) : (
          <Wand2Icon className='size-4' />
        )}
        {t('mask.inpaint')}
      </Button>

      <Separator orientation='vertical' className='mx-0.5 h-4' />

      <Button
        variant='ghost'
        size='xs'
        onClick={render}
        data-testid='toolbar-render'
        disabled={isBusy}
      >
        {isRendering ? (
          <LoaderCircleIcon className='size-4 animate-spin' />
        ) : (
          <TypeIcon className='size-4' />
        )}
        {t('llm.render')}
      </Button>

      <Separator orientation='vertical' className='mx-0.5 h-4' />

      <Button
        variant='ghost'
        size='xs'
        onClick={handleProcessAll}
        data-testid='toolbar-process-all'
        disabled={isBusy}
        className='text-rose-500 hover:text-rose-600'
      >
        {processingAll ? (
          <LoaderCircleIcon className='size-4 animate-spin' />
        ) : (
          <ZapIcon className='size-4' />
        )}
        {t('processing.processAll')}
      </Button>
    </div>
  )
}

type BlockSpeaker = {
  textBlockId: string
  x: number
  y: number
  width: number
  height: number
  name: string | null
  traits: string[]
  confidence: number
  isKnown: boolean
}

type PanelCharacter = {
  name: string
  traits: string[]
  isKnown: boolean
}

type PanelInfo = {
  x: number
  y: number
  width: number
  height: number
  characters: PanelCharacter[]
}

function CharacterDebugPopover() {
  const { t } = useTranslation()
  const currentIndex = useEditorUiStore((s) => s.currentDocumentIndex)
  const totalPages = useEditorUiStore((s) => s.totalPages)
  const [blocks, setBlocks] = useState<BlockSpeaker[] | null>(null)
  const [panels, setPanels] = useState<PanelInfo[]>([])
  const [panelMode, setPanelMode] = useState<'ml' | 'heuristic' | null>(null)
  const [loading, setLoading] = useState(false)
  const [open, setOpen] = useState(false)

  const scan = async () => {
    setLoading(true)
    try {
      const result = await api.scanCharacters(currentIndex)
      setBlocks(result.blocks)
      setPanels(result.panels ?? [])
      setPanelMode(result.panelMode)
      useEditorUiStore.getState().setScanPanels(result.panels ?? [])
    } catch (e) {
      setBlocks([])
      setPanels([])
      useEditorUiStore.getState().setScanPanels([])
      console.error(e)
    } finally {
      setLoading(false)
    }
  }

  const noPage = totalPages === 0

  const blockList = blocks?.map((b, i) => ({ ...b, idx: i + 1 })) ?? []

  // Group blocks by panel using center-point containment
  const panelBlocks: (typeof blockList)[] = panels.map((panel) =>
    blockList.filter((b) => {
      const cx = b.x + b.width / 2
      const cy = b.y + b.height / 2
      return (
        cx >= panel.x &&
        cx <= panel.x + panel.width &&
        cy >= panel.y &&
        cy <= panel.y + panel.height
      )
    }),
  )
  const assignedBlockIds = new Set(panelBlocks.flat().map((b) => b.textBlockId))

  return (
    <Popover open={open} onOpenChange={setOpen}>
      <PopoverTrigger asChild>
        <Button variant='ghost' size='xs'>
          <UsersIcon className='size-4' />
          {t('characters.title')}
        </Button>
      </PopoverTrigger>
      <PopoverContent align='start' className='w-80'>
        <div className='space-y-3 text-sm'>
          <div className='flex items-center justify-between'>
            <p className='text-muted-foreground text-xs font-medium uppercase'>
              {t('characters.title')}
            </p>
            <button
              type='button'
              onClick={() => void scan()}
              disabled={loading || noPage}
              className='text-primary hover:text-primary/80 disabled:text-muted-foreground text-xs font-medium disabled:cursor-not-allowed'
            >
              {loading ? (
                <span className='flex items-center gap-1'>
                  <LoaderCircleIcon className='size-3 animate-spin' />
                  Scanning…
                </span>
              ) : (
                'Scan page'
              )}
            </button>
          </div>

          {panels.length > 0 && (
            <p className='text-muted-foreground text-[11px]'>
              <span className='text-foreground font-mono font-medium'>
                {panels.length}
              </span>{' '}
              panel{panels.length !== 1 ? 's' : ''} detected
              {panelMode && (
                <span
                  className={`ml-1.5 rounded px-1 py-0.5 text-[10px] font-semibold uppercase ${
                    panelMode === 'ml'
                      ? 'bg-green-500/10 text-green-600 dark:text-green-400'
                      : 'bg-amber-500/10 text-amber-600 dark:text-amber-400'
                  }`}
                >
                  {panelMode}
                </span>
              )}
            </p>
          )}

          {noPage && (
            <p className='text-muted-foreground text-xs'>No page open.</p>
          )}

          {!loading && blocks === null && !noPage && (
            <p className='text-muted-foreground text-xs'>
              Click <strong>Scan page</strong> to detect characters and
              speakers.
            </p>
          )}

          {!loading &&
            blocks !== null &&
            blocks.length === 0 &&
            panels.every((p) => p.characters.length === 0) && (
              <p className='text-muted-foreground text-xs'>
                No characters detected. Add characters in{' '}
                <strong>File → Characters</strong> first.
              </p>
            )}

          {panels.length > 0 && (
            <div className='max-h-80 space-y-3 overflow-y-auto'>
              {panels.map((panel, pi) => (
                <div key={pi} className='space-y-1'>
                  <p className='text-muted-foreground text-[10px] font-semibold tracking-wide uppercase'>
                    Panel {pi + 1}
                  </p>

                  {/* All characters in panel */}
                  {panel.characters.length > 0 && (
                    <div className='space-y-0.5'>
                      {panel.characters.map((char, ci) => (
                        <div
                          key={ci}
                          className='bg-muted flex items-center gap-2 rounded px-2 py-1'
                        >
                          <span
                            className={`flex-1 truncate text-xs font-medium ${
                              char.isKnown
                                ? 'text-foreground'
                                : 'text-muted-foreground'
                            }`}
                          >
                            {char.name}
                          </span>
                          {char.traits.length > 0 && (
                            <span className='text-muted-foreground truncate text-[10px]'>
                              {char.traits.slice(0, 2).join(', ')}
                            </span>
                          )}
                          {char.isKnown && (
                            <span className='shrink-0 rounded bg-green-500/10 px-1 py-0.5 text-[9px] font-semibold text-green-600 dark:text-green-400'>
                              known
                            </span>
                          )}
                        </div>
                      ))}
                    </div>
                  )}

                  {/* Speaker blocks in this panel */}
                  {panelBlocks[pi].length > 0 && (
                    <div className='space-y-0.5 pl-2'>
                      {panelBlocks[pi].map((b) => (
                        <div
                          key={b.textBlockId}
                          className='flex items-center gap-2 rounded border-l-2 border-blue-400 px-2 py-0.5'
                        >
                          <span className='text-muted-foreground w-5 font-mono text-[11px]'>
                            #{b.idx}
                          </span>
                          <span className='text-foreground flex-1 truncate text-[11px]'>
                            {b.name}
                          </span>
                          <span
                            className={`shrink-0 font-mono text-[10px] ${
                              b.confidence >= 0.8
                                ? 'text-green-500'
                                : b.confidence >= 0.7
                                  ? 'text-yellow-500'
                                  : 'text-muted-foreground'
                            }`}
                          >
                            {Math.round(b.confidence * 100)}%
                          </span>
                        </div>
                      ))}
                    </div>
                  )}

                  {panel.characters.length === 0 && (
                    <p className='text-muted-foreground px-2 text-[11px]'>
                      No faces detected
                    </p>
                  )}
                </div>
              ))}

              {/* Blocks not matched to any panel */}
              {blockList.filter((b) => !assignedBlockIds.has(b.textBlockId))
                .length > 0 && (
                <div className='space-y-1'>
                  <p className='text-muted-foreground text-[10px] font-semibold tracking-wide uppercase'>
                    Unassigned
                  </p>
                  {blockList
                    .filter((b) => !assignedBlockIds.has(b.textBlockId))
                    .map((b) => (
                      <div
                        key={b.textBlockId}
                        className='bg-muted flex items-center gap-2 rounded px-2 py-1'
                      >
                        <span className='text-muted-foreground w-5 font-mono text-[11px]'>
                          #{b.idx}
                        </span>
                        <span className='text-muted-foreground flex-1 truncate text-xs'>
                          —
                        </span>
                      </div>
                    ))}
                </div>
              )}
            </div>
          )}

          {/* Fallback: no panel data yet but have blocks */}
          {panels.length === 0 && blockList.length > 0 && (
            <div className='max-h-64 space-y-1 overflow-y-auto'>
              {blockList.map((b) => (
                <div
                  key={b.textBlockId}
                  className={`flex items-center gap-2 rounded px-2 py-1 ${
                    b.isKnown ? 'bg-accent' : 'bg-muted'
                  }`}
                >
                  <span className='text-muted-foreground w-5 font-mono text-[11px]'>
                    #{b.idx}
                  </span>
                  <span
                    className={`flex-1 truncate text-xs font-medium ${
                      b.isKnown ? 'text-foreground' : 'text-muted-foreground'
                    }`}
                  >
                    {b.name ?? '—'}
                  </span>
                  <span
                    className={`shrink-0 font-mono text-[11px] ${
                      b.confidence >= 0.8
                        ? 'text-green-500'
                        : b.confidence >= 0.7
                          ? 'text-yellow-500'
                          : 'text-muted-foreground'
                    }`}
                  >
                    {Math.round(b.confidence * 100)}%
                  </span>
                </div>
              ))}
            </div>
          )}
        </div>
      </PopoverContent>
    </Popover>
  )
}

function LlmStatusPopover() {
  const { data: llmModels = [] } = useLlmModelsQuery()
  const llmSelectedModel = useLlmUiStore((state) => state.selectedModel)
  const llmSelectedLanguage = useLlmUiStore((state) => state.selectedLanguage)
  const llmLoading = useLlmUiStore((state) => state.loading)
  const { data: llmReady = false } = useLlmReadyQuery()
  const { llmSetSelectedModel, llmSetSelectedLanguage, llmToggleLoadUnload } =
    useLlmMutations()
  const { t } = useTranslation()
  const apiKeys = usePreferencesStore((state) => state.apiKeys)
  const localLlm = usePreferencesStore((state) => state.localLlm)

  const selectedModelInfo = useMemo(
    () => llmModels.find((m) => m.id === llmSelectedModel),
    [llmModels, llmSelectedModel],
  )
  const isApiModel =
    selectedModelInfo?.source !== 'local' &&
    selectedModelInfo?.source !== undefined
  const apiKeyMissing =
    isApiModel &&
    selectedModelInfo?.source !== 'openai-compatible' &&
    !apiKeys[selectedModelInfo!.source]

  const activeLanguages = useMemo(
    () => selectedModelInfo?.languages ?? [],
    [selectedModelInfo],
  )

  useEffect(() => {
    if (llmModels.length === 0) return
    const hasCurrent = llmModels.some((model) => model.id === llmSelectedModel)
    const nextModel = hasCurrent ? llmSelectedModel : llmModels[0]?.id
    if (!nextModel) return
    const languages =
      llmModels.find((model) => model.id === nextModel)?.languages ?? []
    const nextLanguage =
      llmSelectedLanguage && languages.includes(llmSelectedLanguage)
        ? llmSelectedLanguage
        : languages[0]
    const currentState = useLlmUiStore.getState()
    if (
      currentState.selectedModel === nextModel &&
      currentState.selectedLanguage === nextLanguage
    ) {
      return
    }
    useLlmUiStore.setState((state) => ({
      selectedModel: nextModel,
      selectedLanguage: nextLanguage,
      loading: state.loading,
    }))
  }, [llmModels, llmSelectedLanguage, llmSelectedModel])

  return (
    <Popover>
      <PopoverTrigger asChild>
        <button
          data-testid='llm-trigger'
          data-llm-ready={llmReady ? 'true' : 'false'}
          data-llm-loading={llmLoading ? 'true' : 'false'}
          className={`flex h-6 cursor-pointer items-center gap-1.5 rounded-full px-2.5 text-[11px] font-medium shadow-sm transition hover:opacity-80 ${
            llmReady
              ? 'bg-rose-400 text-white ring-1 ring-rose-400/30'
              : 'bg-muted text-muted-foreground ring-border/50 ring-1'
          }`}
        >
          <motion.span
            className={`size-1.5 rounded-full ${
              llmReady ? 'bg-white' : 'bg-muted-foreground/40'
            }`}
            animate={llmReady ? { opacity: [1, 0.5, 1] } : { opacity: 1 }}
            transition={
              llmReady
                ? { duration: 2, repeat: Infinity, ease: 'easeInOut' }
                : {}
            }
          />
          LLM
          {llmReady && selectedModelInfo?.source === 'openai-compatible' && (
            <span className='max-w-[80px] truncate text-[10px] opacity-80'>
              {selectedModelInfo.id.split(':')[1] ?? selectedModelInfo.id}
            </span>
          )}
        </button>
      </PopoverTrigger>
      <PopoverContent align='end' className='w-72' data-testid='llm-popover'>
        <div className='space-y-3 text-sm'>
          <p className='text-muted-foreground text-xs font-medium uppercase'>
            {t('panels.llm')}
          </p>

          <Select value={llmSelectedModel} onValueChange={llmSetSelectedModel}>
            <SelectTrigger data-testid='llm-model-select' className='w-full'>
              <SelectValue placeholder={t('llm.selectPlaceholder')} />
            </SelectTrigger>
            <SelectContent position='popper'>
              {llmModels.map((model, index) => (
                <SelectItem
                  key={model.id}
                  value={model.id}
                  data-testid={`llm-model-option-${index}`}
                >
                  <span className='flex items-center gap-2'>
                    {model.source === 'openai-compatible' &&
                    model.origin === 'local-llm' ? (
                      <span className='rounded bg-emerald-500/10 px-1 py-0.5 text-[10px] leading-none font-semibold text-emerald-600 uppercase dark:text-emerald-400'>
                        {LOCAL_LLM_PRESET_LABELS[localLlm.preset] ?? 'Local'}
                      </span>
                    ) : model.source === 'openai-compatible' ? (
                      <span className='rounded bg-teal-500/10 px-1 py-0.5 text-[10px] leading-none font-semibold text-teal-600 uppercase dark:text-teal-400'>
                        OpenAI-like
                      </span>
                    ) : model.source !== 'local' ? (
                      <span className='bg-primary/10 text-primary rounded px-1 py-0.5 text-[10px] leading-none font-semibold uppercase'>
                        {getProviderDisplayName(model.source)}
                      </span>
                    ) : null}
                    {model.id.includes(':') ? model.id.split(':')[1] : model.id}
                  </span>
                </SelectItem>
              ))}
            </SelectContent>
          </Select>

          {/* API key warning */}
          {apiKeyMissing && (
            <p className='text-xs text-amber-500'>
              {t('llm.apiKeyMissing', {
                provider: getProviderDisplayName(selectedModelInfo!.source),
              })}
            </p>
          )}

          {/* Loaded model info card */}
          {llmReady && selectedModelInfo?.source === 'openai-compatible' && (
            <div
              className={`rounded-md px-2.5 py-2 text-xs ${
                selectedModelInfo.origin === 'local-llm'
                  ? 'border border-emerald-500/20 bg-emerald-500/5'
                  : 'border border-teal-500/20 bg-teal-500/5'
              }`}
            >
              <div className='flex items-center gap-1.5'>
                <span
                  className={`size-1.5 rounded-full ${
                    selectedModelInfo.origin === 'local-llm'
                      ? 'bg-emerald-500'
                      : 'bg-teal-500'
                  }`}
                />
                <span
                  className={`font-medium ${
                    selectedModelInfo.origin === 'local-llm'
                      ? 'text-emerald-700 dark:text-emerald-400'
                      : 'text-teal-700 dark:text-teal-400'
                  }`}
                >
                  {t('llm.localModelActive')}
                </span>
              </div>
              <p className='text-muted-foreground mt-1'>
                {t('llm.localModelName', {
                  name:
                    selectedModelInfo.id.split(':')[1] ?? selectedModelInfo.id,
                })}
              </p>
              <p className='text-muted-foreground mt-0.5'>
                {selectedModelInfo.origin === 'local-llm'
                  ? (LOCAL_LLM_PRESET_LABELS[localLlm.preset] ?? 'Local')
                  : 'OpenAI-like'}
                {localLlm.temperature != null &&
                  ` · temp ${localLlm.temperature}`}
                {localLlm.maxTokens != null &&
                  ` · ${localLlm.maxTokens} tokens`}
              </p>
            </div>
          )}

          {activeLanguages.length > 0 && (
            <Select
              value={llmSelectedLanguage ?? activeLanguages[0]}
              onValueChange={llmSetSelectedLanguage}
            >
              <SelectTrigger
                data-testid='llm-language-select'
                className='w-full'
              >
                <SelectValue placeholder={t('llm.languagePlaceholder')} />
              </SelectTrigger>
              <SelectContent position='popper'>
                {activeLanguages.map((language, index) => (
                  <SelectItem
                    key={language}
                    value={language}
                    data-testid={`llm-language-option-${index}`}
                  >
                    {t(`llm.languages.${language}`)}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          )}

          <Button
            data-testid='llm-load-toggle'
            data-llm-ready={llmReady ? 'true' : 'false'}
            data-llm-loading={llmLoading ? 'true' : 'false'}
            variant='outline'
            size='sm'
            onClick={llmToggleLoadUnload}
            disabled={!llmSelectedModel || llmLoading}
            className='w-full gap-1.5 text-xs'
          >
            {llmLoading && (
              <LoaderCircleIcon className='size-3.5 animate-spin' />
            )}
            {!llmReady ? t('llm.load') : t('llm.unload')}
          </Button>
        </div>
      </PopoverContent>
    </Popover>
  )
}
