# ScanPro - Document Scanner Mobile UI

A beautiful React/TypeScript mobile UI for the Shadow-Robust Document Scanner.

## Quick Start

```bash
# Navigate to scanpro folder
cd scanpro

# Install dependencies
npm install

# Start development server
npm run dev
```

## Backend Connection

1. Start the Flask API server:
```bash
cd ..
pip install flask flask-cors
python api/app.py --port 5000
```

2. Copy environment config:
```bash
cp .env.example .env.local
```

3. Update `VITE_API_URL` in `.env.local` with your backend URL.

## Project Structure

```
scanpro/
├── screens/            # App screens
│   ├── SplashScreen.tsx
│   ├── OnboardingScreen.tsx
│   ├── CameraScreen.tsx    # Camera/scanning
│   ├── EditScreen.tsx      # Corner adjustment
│   ├── ResultScreen.tsx    # Before/after comparison
│   ├── LibraryScreen.tsx   # Document library
│   ├── ViewerScreen.tsx    # Document viewer
│   ├── SettingsScreen.tsx  # Settings
│   └── ToolsScreen.tsx     # Tools
├── components/
│   └── BottomNav.tsx
├── services/           # [NEW] Backend integration
│   ├── api.ts          # API calls
│   └── scannerStore.ts # State management
├── App.tsx
└── store.ts           # Mock data
```

## API Endpoints Used

| Function | Endpoint | Description |
|----------|----------|-------------|
| `scanDocument()` | POST /api/scan | Full document scan |
| `detectCorners()` | POST /api/detect | Corner detection |
| `enhanceImage()` | POST /api/enhance | Image enhancement |
| `exportToPdf()` | POST /api/export-pdf | PDF export |

## Usage Example

```typescript
import api from './services/api';

// Scan a document
const result = await api.scanDocument(imageBase64, {
  remove_shadows: true,
  enhance: true
});

if (result.success && result.scan) {
  // Display scanned image
  setScannedImage(result.scan);
}
```

## Styling

Uses Tailwind CSS with a custom dark theme:
- Glass effects (backdrop-blur)
- Gradient accents (indigo → cyan)
- Smooth animations
- Material Icons (Google Fonts)
