/**
 * Scanner State Store
 * 
 * Shared state for the scanner across components.
 * Uses a simple reactive pattern for state management.
 */

import { ScanResult } from './api';

export interface ScannedDocument {
    id: string;
    originalImage: string;  // base64
    scannedImage: string;   // base64
    corners: number[][] | null;
    confidence: number;
    timestamp: number;
    title: string;
}

// In-memory store for scanned documents (in real app, use localStorage or DB)
let scannedDocuments: ScannedDocument[] = [];
let listeners: (() => void)[] = [];

export function getScannedDocuments(): ScannedDocument[] {
    return scannedDocuments;
}

export function addScannedDocument(doc: Omit<ScannedDocument, 'id' | 'timestamp'>): ScannedDocument {
    const newDoc: ScannedDocument = {
        ...doc,
        id: Date.now().toString(),
        timestamp: Date.now()
    };
    scannedDocuments = [newDoc, ...scannedDocuments];
    notifyListeners();
    return newDoc;
}

export function removeScannedDocument(id: string): void {
    scannedDocuments = scannedDocuments.filter(d => d.id !== id);
    notifyListeners();
}

export function clearAllDocuments(): void {
    scannedDocuments = [];
    notifyListeners();
}

export function subscribeToDocuments(listener: () => void): () => void {
    listeners.push(listener);
    return () => {
        listeners = listeners.filter(l => l !== listener);
    };
}

function notifyListeners(): void {
    listeners.forEach(l => l());
}

// Current scan state (for passing between screens)
let currentScanState: {
    originalImage: string | null;
    scanResult: ScanResult | null;
    corners: number[][] | null;
} = {
    originalImage: null,
    scanResult: null,
    corners: null
};

export function setCurrentScan(
    originalImage: string,
    result: ScanResult,
    corners: number[][] | null = null
): void {
    currentScanState = {
        originalImage,
        scanResult: result,
        corners: corners || result.corners || null
    };
}

export function getCurrentScan() {
    return currentScanState;
}

export function clearCurrentScan(): void {
    currentScanState = {
        originalImage: null,
        scanResult: null,
        corners: null
    };
}

// Settings store
export interface ScannerSettings {
    removeShadows: boolean;
    enhance: boolean;
    autoCapture: boolean;
    outputFormat: 'jpeg' | 'png';
    saveToGallery: boolean;
}

let settings: ScannerSettings = {
    removeShadows: true,
    enhance: true,
    autoCapture: false,
    outputFormat: 'jpeg',
    saveToGallery: true
};

export function getSettings(): ScannerSettings {
    return { ...settings };
}

export function updateSettings(partial: Partial<ScannerSettings>): void {
    settings = { ...settings, ...partial };
    // Persist to localStorage
    try {
        localStorage.setItem('scanpro_settings', JSON.stringify(settings));
    } catch (e) {
        // Ignore storage errors
    }
}

// Load settings from localStorage on init
try {
    const saved = localStorage.getItem('scanpro_settings');
    if (saved) {
        settings = { ...settings, ...JSON.parse(saved) };
    }
} catch (e) {
    // Ignore storage errors
}
