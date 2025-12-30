import React, { useState } from 'react';
import { ScanResult } from '../services/api';
import { getCurrentScan, addDocument, formatDate, getFileSizeString } from '../store';
import api from '../services/api';

interface ResultScreenProps {
    onAddPage: () => void;
    onRetake: () => void;
    onFinish: () => void;
    scanResult?: ScanResult | null;
}

export const ResultScreen: React.FC<ResultScreenProps> = ({ onAddPage, onRetake, onFinish, scanResult }) => {
    const [sliderPos, setSliderPos] = useState(50);
    const [isSaving, setIsSaving] = useState(false);
    const [saved, setSaved] = useState(false);
    const [docTitle, setDocTitle] = useState(`Scan_${new Date().toISOString().slice(0, 10)}`);

    const handleSliderChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        setSliderPos(Number(e.target.value));
    };

    // Get scan data
    const currentScan = getCurrentScan();
    const originalImage = currentScan.originalImage;
    const scannedImage = scanResult?.scan || currentScan.scannedImage;
    const confidence = scanResult?.confidence || currentScan.confidence || 0;

    const handleSave = async () => {
        if (!scannedImage) return;

        setIsSaving(true);

        try {
            // Save to library
            addDocument({
                title: `${docTitle}.jpg`,
                date: formatDate(new Date()),
                size: getFileSizeString(scannedImage),
                thumbnail: scannedImage,
                type: 'jpg'
            });

            // Also download to device
            api.downloadBase64(scannedImage, `${docTitle}.jpg`);

            setSaved(true);
            setTimeout(() => setSaved(false), 2000);
        } catch (e) {
            console.error('Save failed:', e);
        }

        setIsSaving(false);
    };

    const handleShare = async () => {
        if (!scannedImage) return;

        try {
            const blob = api.base64ToBlob(scannedImage);
            const file = new File([blob], `${docTitle}.jpg`, { type: 'image/jpeg' });

            if (navigator.share) {
                await navigator.share({
                    files: [file],
                    title: docTitle,
                });
            } else {
                // Fallback - download
                api.downloadBase64(scannedImage, `${docTitle}.jpg`);
            }
        } catch (e) {
            console.log('Share failed', e);
        }
    };

    const handleExportPdf = async () => {
        if (!scannedImage) return;

        setIsSaving(true);
        try {
            const result = await api.exportToPdf([scannedImage], { page_size: 'A4' });
            if (result.success && result.pdf) {
                // Save PDF to library
                addDocument({
                    title: `${docTitle}.pdf`,
                    date: formatDate(new Date()),
                    size: getFileSizeString(result.pdf),
                    thumbnail: scannedImage, // Use the scan as thumbnail
                    type: 'pdf',
                    pageCount: 1
                });

                // Download
                api.downloadBase64(result.pdf, `${docTitle}.pdf`);
            }
        } catch (e) {
            console.error('PDF export failed', e);
        }
        setIsSaving(false);
    };

    const handleFinish = () => {
        // Auto-save before finishing if not already saved
        if (!saved && scannedImage) {
            addDocument({
                title: `${docTitle}.jpg`,
                date: formatDate(new Date()),
                size: getFileSizeString(scannedImage),
                thumbnail: scannedImage,
                type: 'jpg'
            });
        }
        onFinish();
    };

    return (
        <div className="bg-background-dark font-display antialiased h-screen w-full flex flex-col overflow-hidden text-white">

            {/* Header */}
            <header className="flex items-center justify-between p-5 pt-12 z-20">
                <button onClick={onRetake} className="flex size-10 items-center justify-center rounded-full hover:bg-white/10 transition-colors">
                    <span className="material-symbols-outlined text-white" style={{ fontSize: 24 }}>arrow_back</span>
                </button>
                <div className="flex-1 text-center">
                    <input
                        type="text"
                        value={docTitle}
                        onChange={(e) => setDocTitle(e.target.value)}
                        className="bg-transparent text-white text-[17px] font-semibold tracking-wide text-center border-none outline-none focus:bg-white/10 rounded-lg px-2 py-1"
                        placeholder="Document name"
                    />
                </div>
                <button className="flex size-10 items-center justify-center rounded-full hover:bg-white/10 transition-colors">
                    <span className="material-symbols-outlined text-white" style={{ fontSize: 24 }}>tune</span>
                </button>
            </header>

            {/* Comparison Area */}
            <main className="flex-1 flex flex-col items-center justify-center w-full relative z-10 px-6 pb-6">
                <div className="group relative w-full h-full max-h-[65vh] rounded-2xl shadow-[0_20px_50px_-20px_rgba(0,0,0,0.5)] overflow-hidden select-none bg-surface-dark border border-white/5">

                    {/* Labels */}
                    <div className="absolute top-4 left-4 z-20">
                        <span className="bg-black/40 backdrop-blur-md text-white/90 text-[10px] font-bold px-3 py-1.5 rounded-full uppercase tracking-wider border border-white/10">Original</span>
                    </div>
                    <div className="absolute top-4 right-4 z-20">
                        <span className="bg-primary/90 backdrop-blur-md text-white text-[10px] font-bold px-3 py-1.5 rounded-full uppercase tracking-wider shadow-lg shadow-primary/20">Enhanced</span>
                    </div>

                    {/* Enhanced Image (Background) */}
                    <div
                        className="absolute inset-0 bg-contain bg-center bg-no-repeat bg-white"
                        style={{
                            backgroundImage: scannedImage ? `url('${scannedImage}')` : undefined,
                            backgroundColor: scannedImage ? 'white' : '#333'
                        }}
                    ></div>

                    {/* Original Image (Clipped) */}
                    <div
                        className="absolute inset-y-0 left-0 overflow-hidden bg-white/5"
                        style={{ width: `${sliderPos}%` }}
                    >
                        <div
                            className="absolute inset-0 h-full w-full bg-contain bg-center bg-no-repeat origin-left filter brightness-90 sepia-[0.1] contrast-75"
                            style={{
                                width: `${10000 / sliderPos}%`,
                                backgroundImage: originalImage ? `url('${originalImage}')` : undefined
                            }}
                        ></div>
                        <div className="absolute inset-y-0 right-0 w-0.5 bg-white/20 shadow-[0_0_10px_rgba(0,0,0,0.5)]"></div>
                    </div>

                    {/* Drag Handle */}
                    <div
                        className="absolute inset-y-0 -ml-[18px] w-9 flex items-center justify-center z-20 pointer-events-none"
                        style={{ left: `${sliderPos}%` }}
                    >
                        <div className="size-9 bg-white rounded-full shadow-[0_2px_10px_rgba(0,0,0,0.4)] flex items-center justify-center">
                            <span className="material-symbols-outlined text-primary" style={{ fontSize: 20 }}>code</span>
                        </div>
                    </div>

                    {/* Range Input */}
                    <input
                        type="range"
                        min="0"
                        max="100"
                        value={sliderPos}
                        onChange={handleSliderChange}
                        className="absolute inset-0 w-full h-full opacity-0 cursor-ew-resize z-30"
                    />

                    {/* Badges */}
                    <div className="absolute bottom-4 right-4 z-30 pointer-events-none flex flex-col gap-2 items-end">
                        <div className="flex items-center gap-x-1.5 rounded-full bg-[#1e1f2e] border border-white/5 pl-2 pr-3 py-1.5 shadow-lg">
                            <span className="material-symbols-outlined text-green-400 filled" style={{ fontSize: 16 }}>check_circle</span>
                            <p className="text-white text-[11px] font-medium leading-none">Shadow Removed</p>
                        </div>
                        {confidence > 0 && (
                            <div className="flex items-center gap-x-1.5 rounded-full bg-[#1e1f2e] border border-white/5 pl-2 pr-3 py-1.5 shadow-lg">
                                <span className="material-symbols-outlined text-blue-400" style={{ fontSize: 16 }}>verified</span>
                                <p className="text-white text-[11px] font-medium leading-none">{Math.round(confidence * 100)}% Confidence</p>
                            </div>
                        )}
                    </div>
                </div>
                <p className="mt-6 text-gray-500 text-[13px] font-medium">Slide to compare</p>
            </main>

            {/* Actions */}
            <div className="w-full pb-8 z-20 flex flex-col gap-8">
                <div className="flex justify-center items-start gap-8">
                    <button onClick={handleShare} className="group flex flex-col items-center gap-2">
                        <div className="size-[3.5rem] rounded-full bg-surface-highlight border border-white/5 flex items-center justify-center group-active:scale-95 transition-transform">
                            <span className="material-symbols-outlined text-white" style={{ fontSize: 24 }}>ios_share</span>
                        </div>
                        <span className="text-[11px] font-medium text-gray-400">Share</span>
                    </button>
                    <button onClick={handleSave} disabled={isSaving} className="group flex flex-col items-center gap-2">
                        <div className={`size-[3.5rem] rounded-full ${saved ? 'bg-green-500/20' : 'bg-surface-highlight'} border border-white/5 flex items-center justify-center group-active:scale-95 transition-transform`}>
                            {isSaving ? (
                                <div className="animate-spin w-5 h-5 border-2 border-white border-t-transparent rounded-full"></div>
                            ) : (
                                <span className={`material-symbols-outlined ${saved ? 'text-green-400' : 'text-white'}`} style={{ fontSize: 24 }}>
                                    {saved ? 'check' : 'save'}
                                </span>
                            )}
                        </div>
                        <span className="text-[11px] font-medium text-gray-400">{saved ? 'Saved!' : 'Save'}</span>
                    </button>
                    <button onClick={handleExportPdf} disabled={isSaving} className="group flex flex-col items-center gap-2">
                        <div className="size-[3.5rem] rounded-full bg-surface-highlight border border-white/5 flex items-center justify-center group-active:scale-95 transition-transform">
                            {isSaving ? (
                                <div className="animate-spin w-5 h-5 border-2 border-white border-t-transparent rounded-full"></div>
                            ) : (
                                <span className="material-symbols-outlined text-white" style={{ fontSize: 24 }}>picture_as_pdf</span>
                            )}
                        </div>
                        <span className="text-[11px] font-medium text-gray-400">PDF</span>
                    </button>
                    <button onClick={onRetake} className="group flex flex-col items-center gap-2">
                        <div className="size-[3.5rem] rounded-full bg-surface-highlight border border-white/5 flex items-center justify-center group-active:scale-95 transition-transform">
                            <span className="material-symbols-outlined text-white" style={{ fontSize: 24 }}>replay</span>
                        </div>
                        <span className="text-[11px] font-medium text-gray-400">Retake</span>
                    </button>
                </div>

                <div className="px-5 flex gap-3">
                    <button onClick={handleFinish} className="flex-1 h-14 bg-surface-highlight border border-white/10 rounded-2xl flex items-center justify-center gap-2 text-white font-bold text-[15px] active:scale-[0.98] transition-all">
                        <span className="material-symbols-outlined" style={{ fontSize: 22 }}>check</span>
                        Done
                    </button>
                    <button onClick={onAddPage} className="flex-1 h-14 bg-primary rounded-2xl flex items-center justify-center gap-2 text-white font-bold text-[15px] shadow-lg shadow-primary/25 active:scale-[0.98] transition-all hover:bg-primary-dark">
                        <span className="material-symbols-outlined" style={{ fontSize: 22 }}>add_a_photo</span>
                        Add Page
                    </button>
                </div>
            </div>
        </div>
    );
};
