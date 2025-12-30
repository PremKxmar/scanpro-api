import React, { useState, useRef } from 'react';
import api from '../services/api';
import { setCurrentScan } from '../store';

interface CameraScreenProps {
    onCapture: (scanResult: any) => void;
    onCancel: () => void;
}

export const CameraScreen: React.FC<CameraScreenProps> = ({ onCapture, onCancel }) => {
    const fileInputRef = useRef<HTMLInputElement>(null);
    const galleryInputRef = useRef<HTMLInputElement>(null);
    const [isScanning, setIsScanning] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [previewImage, setPreviewImage] = useState<string | null>(null);

    const processImage = async (base64Image: string) => {
        setIsScanning(true);
        setError(null);
        setPreviewImage(base64Image);

        try {
            console.log('Sending to API for processing...');

            const result = await api.scanDocument(base64Image, {
                remove_shadows: true,
                enhance: true,
                output_format: 'jpeg'
            });

            console.log('API result:', result);

            if (result.success && result.scan) {
                // Store the scan result
                setCurrentScan(base64Image, result.scan, result.corners || null, result.confidence || 0);

                // Navigate to result screen
                onCapture(result);
            } else {
                setError(result.message || result.error || 'No document detected. Try a clearer image.');

                // Store anyway for retry
                setCurrentScan(base64Image, null, null, 0);
            }
        } catch (err: any) {
            console.error('Scan error:', err);
            setError(`API Error: ${err.message}. Is the backend running?`);
        } finally {
            setIsScanning(false);
        }
    };

    const handleFileSelect = async (e: React.ChangeEvent<HTMLInputElement>) => {
        const file = e.target.files?.[0];
        if (!file) return;

        try {
            const base64 = await api.fileToBase64(file);
            await processImage(base64);
        } catch (err: any) {
            setError('Failed to read image file');
        }

        // Reset input so same file can be selected again
        e.target.value = '';
    };

    const openCamera = () => {
        fileInputRef.current?.click();
    };

    const openGallery = () => {
        galleryInputRef.current?.click();
    };

    return (
        <div className="bg-background-dark text-white font-display overflow-hidden h-screen w-full relative">
            {/* Background / Preview */}
            <div className="absolute inset-0 z-0">
                {previewImage ? (
                    <img
                        src={previewImage}
                        alt="Preview"
                        className="w-full h-full object-cover"
                    />
                ) : (
                    <div className="w-full h-full bg-gradient-to-b from-gray-800 to-gray-900 flex items-center justify-center">
                        <div className="text-center p-6">
                            <span className="material-symbols-outlined text-6xl text-gray-600 mb-4">photo_camera</span>
                            <p className="text-gray-400 text-lg">Tap the camera button below</p>
                            <p className="text-gray-500 text-sm mt-2">to capture a document</p>
                        </div>
                    </div>
                )}
                <div className="absolute inset-0 bg-gradient-to-b from-black/60 via-transparent to-black/70"></div>
            </div>

            {/* Hidden file inputs */}
            <input
                ref={fileInputRef}
                type="file"
                accept="image/*"
                capture="environment"
                className="hidden"
                onChange={handleFileSelect}
            />
            <input
                ref={galleryInputRef}
                type="file"
                accept="image/*"
                className="hidden"
                onChange={handleFileSelect}
            />

            <div className="relative z-10 flex flex-col h-full justify-between">
                {/* Top Status */}
                <div className="pt-12 px-6 pb-4 flex items-start justify-between">
                    <button onClick={onCancel} className="group flex items-center justify-center size-10 rounded-full bg-black/30 backdrop-blur-md border border-white/10 active:bg-white/20 transition-all">
                        <span className="material-symbols-outlined text-white" style={{ fontSize: 24 }}>close</span>
                    </button>

                    <div className="flex flex-col gap-1 items-center bg-black/30 backdrop-blur-md rounded-2xl p-3 border border-white/5">
                        <div className="relative size-10 flex items-center justify-center">
                            {isScanning ? (
                                <div className="animate-spin w-8 h-8 border-2 border-cyan-400 border-t-transparent rounded-full"></div>
                            ) : (
                                <>
                                    <svg className="w-full h-full -rotate-90" viewBox="0 0 36 36">
                                        <path className="text-white/20" d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831" fill="none" stroke="currentColor" strokeWidth="3"></path>
                                        <path className="text-accent-cyan drop-shadow-[0_0_8px_rgba(34,211,238,0.5)]" d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831" fill="none" stroke="currentColor" strokeDasharray="75, 100" strokeLinecap="round" strokeWidth="3"></path>
                                    </svg>
                                    <span className="material-symbols-outlined text-white text-[18px] absolute">document_scanner</span>
                                </>
                            )}
                        </div>
                        <span className="text-[10px] font-medium text-gray-300 uppercase tracking-wider">
                            {isScanning ? 'Processing...' : 'Ready'}
                        </span>
                    </div>

                    <div className="size-12"></div> {/* Spacer */}
                </div>

                {/* Center Scanner Frame */}
                <div className="flex-1 flex flex-col items-center justify-center relative">
                    <div className="relative w-[80%] aspect-[3/4] max-w-sm">
                        {/* Scanning animation */}
                        {isScanning && (
                            <>
                                <div className="absolute left-0 right-0 h-1 bg-cyan-400/50 shadow-[0_0_15px_rgba(34,211,238,0.6)] z-20" style={{
                                    animation: 'scanLine 1.5s ease-in-out infinite'
                                }}></div>
                                <style>{`
                  @keyframes scanLine {
                    0% { top: 5%; opacity: 0; }
                    10% { opacity: 1; }
                    90% { opacity: 1; }
                    100% { top: 95%; opacity: 0; }
                  }
                `}</style>
                            </>
                        )}

                        {/* Corner markers */}
                        <div className={`absolute top-0 left-0 w-10 h-10 border-t-4 border-l-4 ${isScanning ? 'border-green-400' : 'border-cyan-400'} rounded-tl-xl drop-shadow-[0_0_4px_rgba(34,211,238,0.8)] transition-colors`}></div>
                        <div className={`absolute top-0 right-0 w-10 h-10 border-t-4 border-r-4 ${isScanning ? 'border-green-400' : 'border-cyan-400'} rounded-tr-xl drop-shadow-[0_0_4px_rgba(34,211,238,0.8)] transition-colors`}></div>
                        <div className={`absolute bottom-0 right-0 w-10 h-10 border-b-4 border-r-4 ${isScanning ? 'border-green-400' : 'border-cyan-400'} rounded-br-xl drop-shadow-[0_0_4px_rgba(34,211,238,0.8)] transition-colors`}></div>
                        <div className={`absolute bottom-0 left-0 w-10 h-10 border-b-4 border-l-4 ${isScanning ? 'border-green-400' : 'border-cyan-400'} rounded-bl-xl drop-shadow-[0_0_4px_rgba(34,211,238,0.8)] transition-colors`}></div>

                        <div className="absolute inset-0 border border-white/10 rounded-lg"></div>
                    </div>

                    <div className="mt-8">
                        <div className={`flex h-10 items-center justify-center gap-x-2 rounded-full ${error ? 'bg-red-500/20' : 'bg-black/40'} backdrop-blur-lg border ${error ? 'border-red-500/30' : 'border-white/10'} px-5 shadow-lg`}>
                            <span className={`material-symbols-outlined ${error ? 'text-red-400' : 'text-cyan-400'} text-[18px]`}>
                                {isScanning ? 'hourglass_empty' : error ? 'error' : 'center_focus_strong'}
                            </span>
                            <p className={`${error ? 'text-red-300' : 'text-white'} text-sm font-medium leading-normal`}>
                                {isScanning ? 'Processing document...' : error || 'Tap camera to capture'}
                            </p>
                        </div>
                        {error && (
                            <button
                                onClick={() => setError(null)}
                                className="mt-3 px-4 py-2 bg-white/10 rounded-full text-sm text-white"
                            >
                                Dismiss
                            </button>
                        )}
                    </div>
                </div>

                {/* Bottom Bar */}
                <div className="w-full glass-panel rounded-t-[2.5rem] pb-8 pt-6 px-8 transition-transform duration-500 ease-out">
                    <div className="flex items-center justify-between max-w-md mx-auto relative">
                        {/* Gallery button */}
                        <button onClick={openGallery} className="flex flex-col items-center gap-1 group relative">
                            <div className="size-12 rounded-2xl bg-white/10 overflow-hidden border border-white/20 flex items-center justify-center group-active:scale-95 transition-transform">
                                <span className="material-symbols-outlined text-white text-[20px]">photo_library</span>
                            </div>
                            <span className="text-[10px] font-medium text-white/60">Gallery</span>
                        </button>

                        {/* Capture button */}
                        <div className="relative group cursor-pointer -mt-6">
                            <div className={`absolute inset-0 rounded-full ${isScanning ? 'bg-green-500/30' : 'bg-primary/30'} blur-xl`}></div>
                            <div className="size-20 rounded-full border-[3px] border-white/20 flex items-center justify-center bg-black/20 backdrop-blur-sm relative z-10">
                                <button
                                    onClick={openCamera}
                                    disabled={isScanning}
                                    className={`size-16 rounded-full ${isScanning ? 'bg-green-500' : 'bg-gradient-to-br from-[#6467f2] to-[#22d3ee]'} shadow-[0_0_20px_rgba(100,103,242,0.6)] active:scale-90 transition-all duration-200 flex items-center justify-center relative overflow-hidden disabled:opacity-50`}
                                >
                                    <div className="absolute top-0 left-0 w-full h-1/2 bg-white/10 rounded-t-full"></div>
                                    {isScanning ? (
                                        <div className="animate-spin w-6 h-6 border-2 border-white border-t-transparent rounded-full"></div>
                                    ) : (
                                        <span className="material-symbols-outlined text-white drop-shadow-md text-3xl">photo_camera</span>
                                    )}
                                </button>
                            </div>
                        </div>

                        {/* Cancel button */}
                        <button onClick={onCancel} className="flex flex-col items-center gap-1 group">
                            <div className="size-12 rounded-full bg-transparent hover:bg-white/10 border border-transparent hover:border-white/10 flex items-center justify-center transition-all group-active:scale-95">
                                <span className="material-symbols-outlined text-white text-[28px]">close</span>
                            </div>
                            <span className="text-[10px] font-medium text-white/60">Cancel</span>
                        </button>
                    </div>
                    <div className="text-center mt-4">
                        <p className="text-white/40 text-xs font-medium uppercase tracking-widest">
                            {isScanning ? 'Processing...' : 'Document Scanner'}
                        </p>
                    </div>
                </div>
            </div>
        </div>
    );
};
