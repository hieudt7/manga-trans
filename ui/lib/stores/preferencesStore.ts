'use client'

import { create } from 'zustand'
import { persist } from 'zustand/middleware'

export type LocalLlmConfig = {
  preset: 'ollama' | 'lmstudio' | 'custom'
  baseUrl: string
  apiKey: string
  modelName: string
  temperature: number | null
  maxTokens: number | null
  customSystemPrompt: string
  storyContext: string
  targetLanguage: string
}

type PreferencesState = {
  brushConfig: {
    size: number
    color: string
  }
  setBrushConfig: (config: Partial<PreferencesState['brushConfig']>) => void
  fontFamily?: string
  setFontFamily: (font?: string) => void
  apiKeys: Record<string, string>
  setApiKey: (provider: string, key: string) => void
  /// Keys the backend can authenticate with per provider, including any loaded
  /// from a key file — a provider can be usable with an empty Settings field.
  providerKeyCounts: Record<string, number>
  setProviderKeyCount: (provider: string, count: number) => void
  /// 1-based key to start from, for pools whose earlier keys are already spent.
  providerKeyStartIndex: Record<string, number>
  setProviderKeyStartIndex: (provider: string, index: number) => void
  providerBaseUrls: Record<string, string>
  setProviderBaseUrl: (provider: string, url: string) => void
  providerModelNames: Record<string, string>
  setProviderModelName: (provider: string, name: string) => void
  providerStoryContexts: Record<string, string>
  setProviderStoryContext: (provider: string, context: string) => void
  providerCustomPrompts: Record<string, string>
  setProviderCustomPrompt: (provider: string, prompt: string) => void
  openAiCompatibleConfigVersion: number
  localLlm: LocalLlmConfig
  setLocalLlm: (config: Partial<LocalLlmConfig>) => void
  resetPreferences: () => void
}

const initialLocalLlm: LocalLlmConfig = {
  preset: 'ollama',
  baseUrl: 'http://localhost:11434/v1',
  apiKey: '',
  modelName: '',
  temperature: null,
  maxTokens: null,
  customSystemPrompt: '',
  storyContext: '',
  targetLanguage: 'en-US',
}

const initialPreferences = {
  brushConfig: {
    size: 36,
    color: '#ffffff',
  },
  fontFamily: undefined as string | undefined,
  apiKeys: {} as Record<string, string>,
  providerKeyCounts: {} as Record<string, number>,
  providerKeyStartIndex: {} as Record<string, number>,
  providerBaseUrls: {} as Record<string, string>,
  providerModelNames: {} as Record<string, string>,
  providerStoryContexts: {} as Record<string, string>,
  providerCustomPrompts: {} as Record<string, string>,
  openAiCompatibleConfigVersion: 0,
  localLlm: initialLocalLlm,
}

export const usePreferencesStore = create<PreferencesState>()(
  persist(
    (set) => ({
      ...initialPreferences,
      setBrushConfig: (config) =>
        set((state) => ({
          brushConfig: {
            ...state.brushConfig,
            ...config,
          },
        })),
      setFontFamily: (font) => set({ fontFamily: font }),
      setProviderKeyStartIndex: (provider, index) =>
        set((state) => ({
          providerKeyStartIndex: {
            ...state.providerKeyStartIndex,
            [provider]: index,
          },
        })),
      setProviderKeyCount: (provider, count) =>
        set((state) => ({
          providerKeyCounts: { ...state.providerKeyCounts, [provider]: count },
        })),
      setApiKey: (provider, key) =>
        set((state) => ({
          apiKeys: { ...state.apiKeys, [provider]: key },
          openAiCompatibleConfigVersion:
            provider === 'openai-compatible'
              ? state.openAiCompatibleConfigVersion + 1
              : state.openAiCompatibleConfigVersion,
        })),
      setProviderBaseUrl: (provider, url) =>
        set((state) => ({
          providerBaseUrls: {
            ...state.providerBaseUrls,
            [provider]: url,
          },
          openAiCompatibleConfigVersion:
            provider === 'openai-compatible'
              ? state.openAiCompatibleConfigVersion + 1
              : state.openAiCompatibleConfigVersion,
        })),
      setProviderModelName: (provider, name) =>
        set((state) => ({
          providerModelNames: {
            ...state.providerModelNames,
            [provider]: name,
          },
          openAiCompatibleConfigVersion:
            provider === 'openai-compatible'
              ? state.openAiCompatibleConfigVersion + 1
              : state.openAiCompatibleConfigVersion,
        })),
      setProviderStoryContext: (provider, context) =>
        set((state) => ({
          providerStoryContexts: {
            ...state.providerStoryContexts,
            [provider]: context,
          },
        })),
      setProviderCustomPrompt: (provider, prompt) =>
        set((state) => ({
          providerCustomPrompts: {
            ...state.providerCustomPrompts,
            [provider]: prompt,
          },
        })),
      setLocalLlm: (config) =>
        set((state) => ({
          localLlm: { ...state.localLlm, ...config },
          openAiCompatibleConfigVersion:
            state.openAiCompatibleConfigVersion + 1,
        })),
      resetPreferences: () => set({ ...initialPreferences }),
    }),
    {
      name: 'koharu-config',
      partialize: (state) => ({
        brushConfig: state.brushConfig,
        fontFamily: state.fontFamily,
        providerBaseUrls: state.providerBaseUrls,
        providerModelNames: state.providerModelNames,
        providerStoryContexts: state.providerStoryContexts,
        providerCustomPrompts: state.providerCustomPrompts,
        // Persisted: the user picks it once per day. providerKeyCounts is not —
        // it is re-read from the backend on every connect.
        providerKeyStartIndex: state.providerKeyStartIndex,
        localLlm: state.localLlm,
      }),
    },
  ),
)
