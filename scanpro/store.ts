/**
 * Document Storage - Real Storage with localStorage Persistence
 * 
 * This replaces all mock data with real storage functionality.
 * Documents are saved to localStorage and persist across sessions.
 */

import { DocumentItem } from './types';

// Storage key
const STORAGE_KEY = 'scanpro_documents';
const SETTINGS_KEY = 'scanpro_settings';

// In-memory cache
let documentsCache: DocumentItem[] | null = null;

/**
 * Load documents from localStorage
 */
export function loadDocuments(): DocumentItem[] {
  if (documentsCache !== null) {
    return documentsCache;
  }

  try {
    const stored = localStorage.getItem(STORAGE_KEY);
    if (stored) {
      documentsCache = JSON.parse(stored);
      return documentsCache!;
    }
  } catch (e) {
    console.error('Failed to load documents:', e);
  }

  documentsCache = [];
  return [];
}

/**
 * Save documents to localStorage
 */
function saveDocuments(documents: DocumentItem[]): void {
  documentsCache = documents;
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(documents));
  } catch (e) {
    console.error('Failed to save documents:', e);
  }
}

/**
 * Add a new scanned document
 */
export function addDocument(doc: Omit<DocumentItem, 'id'>): DocumentItem {
  const documents = loadDocuments();

  const newDoc: DocumentItem = {
    ...doc,
    id: `doc_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`
  };

  // Add to beginning of list
  const updated = [newDoc, ...documents];
  saveDocuments(updated);

  return newDoc;
}

/**
 * Remove a document by ID
 */
export function removeDocument(id: string): void {
  const documents = loadDocuments();
  const updated = documents.filter(d => d.id !== id);
  saveDocuments(updated);
}

/**
 * Get a document by ID
 */
export function getDocument(id: string): DocumentItem | undefined {
  const documents = loadDocuments();
  return documents.find(d => d.id === id);
}

/**
 * Update a document
 */
export function updateDocument(id: string, updates: Partial<DocumentItem>): void {
  const documents = loadDocuments();
  const index = documents.findIndex(d => d.id === id);
  if (index !== -1) {
    documents[index] = { ...documents[index], ...updates };
    saveDocuments(documents);
  }
}

/**
 * Clear all documents
 */
export function clearAllDocuments(): void {
  saveDocuments([]);
}

/**
 * Get document count
 */
export function getDocumentCount(): number {
  return loadDocuments().length;
}

// ============================================================================
// Settings
// ============================================================================

export interface AppSettings {
  scanQuality: 'low' | 'medium' | 'high';
  autoEnhance: boolean;
  saveToGallery: boolean;
  cloudSync: boolean;
  theme: 'dark' | 'light' | 'system';
  removeShadows: boolean;
}

const defaultSettings: AppSettings = {
  scanQuality: 'high',
  autoEnhance: true,
  saveToGallery: false,
  cloudSync: false,
  theme: 'dark',
  removeShadows: true
};

export function loadSettings(): AppSettings {
  try {
    const stored = localStorage.getItem(SETTINGS_KEY);
    if (stored) {
      return { ...defaultSettings, ...JSON.parse(stored) };
    }
  } catch (e) {
    console.error('Failed to load settings:', e);
  }
  return defaultSettings;
}

export function saveSettings(settings: Partial<AppSettings>): AppSettings {
  const current = loadSettings();
  const updated = { ...current, ...settings };
  try {
    localStorage.setItem(SETTINGS_KEY, JSON.stringify(updated));
  } catch (e) {
    console.error('Failed to save settings:', e);
  }
  return updated;
}

// ============================================================================
// Current Scan State (for passing between screens)
// ============================================================================

interface CurrentScanState {
  originalImage: string | null;
  scannedImage: string | null;
  corners: number[][] | null;
  confidence: number;
}

let currentScan: CurrentScanState = {
  originalImage: null,
  scannedImage: null,
  corners: null,
  confidence: 0
};

export function setCurrentScan(
  original: string,
  scanned: string | null,
  corners: number[][] | null,
  confidence: number
): void {
  currentScan = {
    originalImage: original,
    scannedImage: scanned,
    corners,
    confidence
  };
}

export function getCurrentScan(): CurrentScanState {
  return currentScan;
}

export function clearCurrentScan(): void {
  currentScan = {
    originalImage: null,
    scannedImage: null,
    corners: null,
    confidence: 0
  };
}

// ============================================================================
// Helper: Calculate file size string from base64
// ============================================================================

export function getFileSizeString(base64: string): string {
  // Base64 is ~4/3 the size of the original data
  const bytes = Math.round((base64.length * 3) / 4);

  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

// ============================================================================
// Helper: Format date
// ============================================================================

export function formatDate(timestamp: number | Date): string {
  const date = new Date(timestamp);
  const options: Intl.DateTimeFormatOptions = {
    month: 'short',
    day: 'numeric',
    year: 'numeric'
  };
  return date.toLocaleDateString('en-US', options);
}

// For backwards compatibility - empty array instead of mock data
export const mockDocs: DocumentItem[] = [];
