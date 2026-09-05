'use client'

import { useEffect, useRef, useState, type ReactNode } from 'react'
import { I18nextProvider } from 'react-i18next'
import { ThemeProvider } from 'next-themes'
import { QueryClientProvider, useQueryClient } from '@tanstack/react-query'
import { TooltipProvider } from '@/components/ui/tooltip'
import {
  ProgressBarStatus,
  getCurrentWindow,
  listen,
  subscribeDocumentChanged,
  subscribeDocumentsChanged,
  subscribeJobChanged,
  subscribeLlmChanged,
  subscribeSnapshot,
} from '@/lib/backend'
import i18n from '@/lib/i18n'
import { getQueryClient } from '@/lib/query/client'
import { queryKeys } from '@/lib/query/keys'
import { useApiKeyQuery, useDocumentsCountQuery } from '@/lib/query/hooks'
import { useDownloadStore } from '@/lib/downloads'
import { useEditorUiStore } from '@/lib/stores/editorUiStore'
import { useLlmUiStore } from '@/lib/stores/llmUiStore'
import { useOperationStore } from '@/lib/stores/operationStore'
import { usePreferencesStore } from '@/lib/stores/preferencesStore'
import { isTauri } from '@/lib/backend'
import { useRpcConnection } from '@/hooks/useRpcConnection'
import type {
  DocumentSummary,
  JobState,
  LlmState,
  SnapshotEvent,
} from '@/lib/protocol'

function ProvidersBootstrap({ children }: { children: ReactNode }) {
  const queryClient = useQueryClient()
  const hasConnectedRef = useRef(false)
  const setTotalPages = useEditorUiStore((state) => state.setTotalPages)
  const setApiKey = usePreferencesStore((state) => state.setApiKey)
  const setProviderKeyCount = usePreferencesStore(
    (state) => state.setProviderKeyCount,
  )
  const rpcConnected = useRpcConnection()
  // The keyring and the key file live in the backend process, reached over
  // HTTP — so an RPC connection is the only requirement. Gating on isTauri()
  // meant that running the UI in a browser against the same backend reported
  // every provider as having no key.
  const shouldQueryApiKeys = rpcConnected
  const { data: documentsCount } = useDocumentsCountQuery(rpcConnected)
  const openAiApiKeyQuery = useApiKeyQuery('openai', shouldQueryApiKeys)
  const openAiCompatibleApiKeyQuery = useApiKeyQuery(
    'openai-compatible',
    shouldQueryApiKeys,
  )
  const geminiApiKeyQuery = useApiKeyQuery('gemini', shouldQueryApiKeys)
  const grokApiKeyQuery = useApiKeyQuery('grok', shouldQueryApiKeys)
  const claudeApiKeyQuery = useApiKeyQuery('claude', shouldQueryApiKeys)
  const deepSeekApiKeyQuery = useApiKeyQuery('deepseek', shouldQueryApiKeys)

  const applyDocumentsSnapshot = (documents: DocumentSummary[]) => {
    const count = documents.length
    useEditorUiStore.setState((state) => ({
      totalPages: count,
      currentDocumentIndex:
        count === 0 ? 0 : Math.min(state.currentDocumentIndex, count - 1),
      selectedBlockIndex: count === 0 ? undefined : state.selectedBlockIndex,
      documentsVersion: state.documentsVersion + 1,
    }))
    queryClient.setQueryData(queryKeys.documents.count, count)
    queryClient.invalidateQueries({
      queryKey: queryKeys.documents.currentRoot,
    })
    queryClient.invalidateQueries({
      queryKey: queryKeys.documents.thumbnailRoot,
    })
  }

  const applyLlmSnapshot = (llm: LlmState) => {
    const selectedModel = useLlmUiStore.getState().selectedModel
    const isReady =
      llm.status === 'ready' &&
      (!selectedModel || !llm.modelId || llm.modelId === selectedModel)
    queryClient.setQueryData(queryKeys.llm.ready(selectedModel), isReady)
    useLlmUiStore.getState().setLoading(llm.status === 'loading')

    if (llm.status !== 'loading') {
      const operation = useOperationStore.getState().operation
      if (operation?.type === 'llm-load') {
        useOperationStore.getState().finishOperation()
        getCurrentWindow()
          .setProgressBar({
            status: ProgressBarStatus.None,
            progress: 0,
          })
          .catch(() => {})
      }
    }
  }

  const updatePipelineUi = (job: JobState | null) => {
    const operationStore = useOperationStore.getState()

    if (!job) {
      return
    }

    if (job.status === 'running') {
      const isSingleDoc = job.totalDocuments <= 1
      operationStore.updateOperation({
        step: job.step ?? undefined,
        current: isSingleDoc
          ? job.currentStepIndex
          : job.currentDocument +
            (job.totalSteps > 0 ? job.currentStepIndex / job.totalSteps : 0),
        total: isSingleDoc ? job.totalSteps : job.totalDocuments,
      })

      getCurrentWindow()
        .setProgressBar({
          status: ProgressBarStatus.Normal,
          progress: job.overallPercent,
        })
        .catch(() => {})
      return
    }

    operationStore.updateOperation({
      current: operationStore.operation?.total,
      total: operationStore.operation?.total,
    })

    getCurrentWindow()
      .setProgressBar({ status: ProgressBarStatus.Normal, progress: 100 })
      .catch(() => {})

    queryClient.invalidateQueries({
      queryKey: queryKeys.documents.currentRoot,
    })
    queryClient.invalidateQueries({
      queryKey: queryKeys.documents.thumbnailRoot,
    })

    setTimeout(() => {
      useOperationStore.getState().finishOperation()
      getCurrentWindow()
        .setProgressBar({
          status: ProgressBarStatus.None,
          progress: 0,
        })
        .catch(() => {})
    }, 1000)
  }

  useEffect(() => {
    if (!rpcConnected) return

    if (hasConnectedRef.current) {
      queryClient.invalidateQueries({ type: 'active' })
      return
    }

    hasConnectedRef.current = true
  }, [queryClient, rpcConnected])

  useEffect(() => {
    if (typeof documentsCount === 'number') {
      setTotalPages(documentsCount)
    }
  }, [documentsCount, setTotalPages])

  useEffect(() => {
    if (openAiApiKeyQuery.status === 'success') {
      setApiKey('openai', openAiApiKeyQuery.data?.apiKey ?? '')
      setProviderKeyCount('openai', openAiApiKeyQuery.data?.availableKeys ?? 0)
    }
  }, [
    openAiApiKeyQuery.data,
    openAiApiKeyQuery.status,
    setApiKey,
    setProviderKeyCount,
  ])

  useEffect(() => {
    if (openAiCompatibleApiKeyQuery.status === 'success') {
      setApiKey(
        'openai-compatible',
        openAiCompatibleApiKeyQuery.data?.apiKey ?? '',
      )
      setProviderKeyCount(
        'openai-compatible',
        openAiCompatibleApiKeyQuery.data?.availableKeys ?? 0,
      )
    }
  }, [
    openAiCompatibleApiKeyQuery.data,
    openAiCompatibleApiKeyQuery.status,
    setApiKey,
    setProviderKeyCount,
  ])

  useEffect(() => {
    if (geminiApiKeyQuery.status === 'success') {
      setApiKey('gemini', geminiApiKeyQuery.data?.apiKey ?? '')
      setProviderKeyCount('gemini', geminiApiKeyQuery.data?.availableKeys ?? 0)
    }
  }, [
    geminiApiKeyQuery.data,
    geminiApiKeyQuery.status,
    setApiKey,
    setProviderKeyCount,
  ])

  useEffect(() => {
    if (grokApiKeyQuery.status === 'success') {
      setApiKey('grok', grokApiKeyQuery.data?.apiKey ?? '')
      setProviderKeyCount('grok', grokApiKeyQuery.data?.availableKeys ?? 0)
    }
  }, [
    grokApiKeyQuery.data,
    grokApiKeyQuery.status,
    setApiKey,
    setProviderKeyCount,
  ])

  useEffect(() => {
    if (claudeApiKeyQuery.status === 'success') {
      setApiKey('claude', claudeApiKeyQuery.data?.apiKey ?? '')
      setProviderKeyCount('claude', claudeApiKeyQuery.data?.availableKeys ?? 0)
    }
  }, [
    claudeApiKeyQuery.data,
    claudeApiKeyQuery.status,
    setApiKey,
    setProviderKeyCount,
  ])

  useEffect(() => {
    if (deepSeekApiKeyQuery.status === 'success') {
      setApiKey('deepseek', deepSeekApiKeyQuery.data?.apiKey ?? '')
      setProviderKeyCount(
        'deepseek',
        deepSeekApiKeyQuery.data?.availableKeys ?? 0,
      )
    }
  }, [
    deepSeekApiKeyQuery.data,
    deepSeekApiKeyQuery.status,
    setApiKey,
    setProviderKeyCount,
  ])

  useEffect(() => {
    let unlisten: (() => void) | undefined
    ;(async () => {
      try {
        unlisten = await listen<number>('documents:opened', (event) => {
          const count = event.payload ?? 0
          setTotalPages(count)
          queryClient.setQueryData(queryKeys.documents.count, count)
          queryClient.invalidateQueries({
            queryKey: queryKeys.documents.currentRoot,
          })
          queryClient.invalidateQueries({
            queryKey: queryKeys.documents.thumbnailRoot,
          })
        })
      } catch (_) {}
    })()

    const unsubscribeSnapshot = subscribeSnapshot((payload: SnapshotEvent) => {
      applyDocumentsSnapshot(payload.documents)
      applyLlmSnapshot(payload.llm)
      const pipelineJob =
        payload.jobs.find(
          (job) => job.kind === 'pipeline' || job.kind === 'pipeline-folder',
        ) ?? null
      updatePipelineUi(pipelineJob)
    })

    const unsubscribeDocuments = subscribeDocumentsChanged((payload) => {
      applyDocumentsSnapshot(payload.documents)
    })

    const unsubscribeDocument = subscribeDocumentChanged(() => {
      queryClient.invalidateQueries({
        queryKey: queryKeys.documents.currentRoot,
      })
      queryClient.invalidateQueries({
        queryKey: queryKeys.documents.thumbnailRoot,
      })
    })

    const unsubscribeJobs = subscribeJobChanged((job) => {
      if (job.kind !== 'pipeline' && job.kind !== 'pipeline-folder') return
      updatePipelineUi(job)
      queryClient.invalidateQueries({
        queryKey: queryKeys.documents.currentRoot,
      })
      queryClient.invalidateQueries({
        queryKey: queryKeys.documents.thumbnailRoot,
      })
    })

    const unsubscribeLlm = subscribeLlmChanged((llm) => {
      applyLlmSnapshot(llm)
    })

    return () => {
      unlisten?.()
      unsubscribeSnapshot()
      unsubscribeDocuments()
      unsubscribeDocument()
      unsubscribeJobs()
      unsubscribeLlm()
    }
  }, [queryClient, setTotalPages])

  return children
}

export function Providers({ children }: { children: ReactNode }) {
  const [mounted, setMounted] = useState(false)
  const queryClient = getQueryClient()
  const ensureDownloadSubscribed = useDownloadStore(
    (state) => state.ensureSubscribed,
  )

  useEffect(() => {
    ensureDownloadSubscribed()
  }, [ensureDownloadSubscribed])

  useEffect(() => {
    setMounted(true)

    const handleLanguageChange = (lng: string) => {
      document.documentElement.lang = lng
    }

    handleLanguageChange(i18n.language)
    i18n.on('languageChanged', handleLanguageChange)
    return () => {
      i18n.off('languageChanged', handleLanguageChange)
    }
  }, [])

  if (!mounted) return null

  return (
    <QueryClientProvider client={queryClient}>
      <ProvidersBootstrap>
        <I18nextProvider i18n={i18n}>
          <ThemeProvider attribute='class' defaultTheme='system' enableSystem>
            <TooltipProvider delayDuration={0}>{children}</TooltipProvider>
          </ThemeProvider>
        </I18nextProvider>
      </ProvidersBootstrap>
    </QueryClientProvider>
  )
}

export default Providers
