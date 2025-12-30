# Document Scanner Mobile App - Google Stitch / AI Studio Prompt

> **Copy this entire document into Google Stitch to generate a stunning mobile UI, then export to AI Studio for final code generation.**

---

## App Overview

Create a **premium, modern document scanner mobile app** with a beautiful dark-themed UI featuring glassmorphism, smooth animations, and 3D effects. The app should look like a top-tier product similar to the quality of apps from Apple, Google, or premium startups.

---

## Design System

### Color Palette
```
Primary:        #6366F1 (Vibrant Indigo)
Primary Light:  #818CF8
Primary Dark:   #4F46E5
Accent:         #22D3EE (Cyan)
Success:        #10B981 (Emerald)
Warning:        #F59E0B (Amber)
Error:          #EF4444 (Red)

Background:     #0F0F23 (Deep Dark)
Surface:        #1A1A2E (Card Background)
Surface Light:  #25253A
Border:         rgba(255, 255, 255, 0.1)

Text Primary:   #FFFFFF
Text Secondary: #A0A0B2
Text Muted:     #6B6B7B
```

### Typography
- **Font Family**: Inter or SF Pro Display
- **Headers**: Bold, large with subtle gradient text effects
- **Body**: Regular weight with comfortable line spacing
- **Labels**: Medium weight, slightly muted color

### Glass Effect
All cards and surfaces should have:
```css
background: rgba(26, 26, 46, 0.7);
backdrop-filter: blur(20px);
border: 1px solid rgba(255, 255, 255, 0.08);
border-radius: 24px;
```

---

## Screens to Generate

### 1. Splash Screen
**Design:**
- Full dark background with animated gradient mesh
- Large app logo in center with subtle glow effect
- App name "ScanPro" with gradient text (indigo to cyan)
- Smooth fade-in animation
- Tagline: "Scan smarter. Shadow-free."

---

### 2. Home Screen (Camera View)
**Design:**
- Full-screen camera preview with dark overlay at edges
- **Document Detection Overlay**: Animated corner markers that glow when document is detected
  - Corner markers: 4 L-shaped brackets with pulsing cyan glow
  - When document detected: corners turn green with ripple animation
- **Bottom Action Bar**: Glass panel with rounded corners sliding up from bottom
  - Large circular capture button (gradient indigo to cyan with glow)
  - Gallery button (left) - open recent scans
  - Settings button (right) - gear icon
  - Flash toggle button
- **Top Status Bar**: Shows detection confidence with animated progress arc
- **Floating Hint**: "Position document within frame" with subtle animation

**Animations:**
- Corner markers animate smoothly to detected document edges
- Capture button pulses subtly when ready
- Success haptic and flash effect on capture

---

### 3. Preview & Edit Screen
**Design:**
- Shows captured/scanned document on dark background
- **Corner Adjustment**: Draggable corner handles with glow halos
  - Lines connecting corners with gradient stroke
  - Handles have magnetic snap feel
- **Bottom Toolbar**: Glass bar with edit options
  - Crop/Adjust corners
  - Rotate (90°)
  - Filter options (Auto, B&W, Color, Magic cleanup)
  - Shadow removal toggle
- **Top Bar**: Back arrow, "Edit" title, "Done" button (gradient)

**Animations:**
- Smooth transitions when adjusting corners
- Filter previews with crossfade
- Button press scales with spring physics

---

### 4. Scan Result Screen
**Design:**
- Before/After comparison slider (swipe to compare original vs scanned)
- Scanned document displayed in a floating card with shadow
- **Action Buttons** (horizontal scroll):
  - Share (icon + label)
  - Save to Gallery
  - Export PDF
  - Copy Text (OCR)
  - Retake
- **Quality Badge**: Shows "HD Quality" or "Shadow Removed" with checkmark
- **Bottom**: "Add Another Page" button for multi-page documents

---

### 5. Documents Library Screen
**Design:**
- Grid view of scanned documents (2 columns)
- Each card shows:
  - Thumbnail with rounded corners
  - Document name
  - Date scanned
  - Page count badge
- **Search bar** at top with glass effect
- **Floating Action Button**: Large "+" button with gradient and shadow
- **Empty state**: Illustration with "No scans yet" message

---

### 6. Document Viewer Screen
**Design:**
- Full document view with pinch-to-zoom
- Page indicator dots at bottom for multi-page docs
- **Quick Actions Bar** (floating at bottom):
  - Share
  - Edit
  - Delete
  - Export
- Swipe between pages with smooth parallax effect

---

### 7. Settings Screen
**Design:**
- List of settings in glass cards
- Settings groups:
  - **Scan Quality**: Auto, High, Maximum
  - **Auto-enhance**: Toggle
  - **Shadow Removal**: Toggle (default ON)
  - **Save Location**: Cloud / Device
  - **Theme**: Dark (default) / Light / Auto
- **About section**: App version, rate app, privacy policy
- Toggles should have smooth animated switches with gradient

---

### 8. Export/Share Sheet
**Design:**
- Bottom sheet sliding up with glass background
- Export format options as selectable chips:
  - PDF
  - PNG
  - JPEG
  - Text (OCR)
- Quality slider
- Share to apps grid
- "Export" primary button with gradient

---

## Component Specifications

### Capture Button
```
Size: 72px diameter
Background: Linear gradient 135deg from #6366F1 to #22D3EE
Shadow: 0 8px 32px rgba(99, 102, 241, 0.4)
Border: 3px solid white
Inner circle: 56px white
Animation: Scale 0.95 on press, pulse glow when ready
```

### Glass Card
```
Background: rgba(26, 26, 46, 0.7)
Backdrop blur: 20px
Border: 1px solid rgba(255, 255, 255, 0.08)
Border radius: 24px
Shadow: 0 8px 32px rgba(0, 0, 0, 0.3)
```

### Primary Button
```
Background: Linear gradient 135deg #6366F1 to #4F46E5
Border radius: 16px
Padding: 16px 32px
Text: White, bold, 16px
Shadow: 0 4px 16px rgba(99, 102, 241, 0.3)
Hover/Press: Scale 0.98, brightness increase
```

### Corner Detection Overlay
```
Each corner: L-shaped bracket, 3px stroke
Default color: rgba(255, 255, 255, 0.5)
Detected color: #22D3EE with glow
Success color: #10B981 with pulse animation
Transition: 200ms ease-out
```

---

## Animations to Include

1. **Page Transitions**: Smooth slide with slight parallax
2. **Corner Detection**: Morphing animation as corners adjust
3. **Capture Flash**: Brief white overlay fade
4. **Button Interactions**: Spring physics scale animation
5. **Loading States**: Skeleton shimmer with gradient
6. **Success States**: Confetti or checkmark burst animation
7. **Pull to Refresh**: Custom animated scanner icon
8. **Tab Switches**: Crossfade with subtle slide

---

## API Integration Points

The UI connects to a Flask backend. Generate placeholder functions for these API calls:

```javascript
// API Configuration
const API_BASE_URL = 'http://YOUR_SERVER:5000';

// Scan document - POST /api/scan
async function scanDocument(imageBase64, options = {}) {
  // Send image to backend, receive scanned result
  // options: { remove_shadows: true, enhance: true }
  // Returns: { success, scan, corners, confidence }
}

// Detect corners only - POST /api/detect  
async function detectCorners(imageBase64) {
  // Quick corner detection for preview
  // Returns: { success, detected, corners, confidence }
}

// Enhance image - POST /api/enhance
async function enhanceImage(imageBase64, options = {}) {
  // Apply shadow removal and sharpening
  // Returns: { success, enhanced }
}

// Export to PDF - POST /api/export-pdf
async function exportToPdf(images, options = {}) {
  // Export scanned images to PDF
  // Returns: { success, pdf (base64) }
}

// Health check - GET /api/health
async function checkHealth() {
  // Verify backend is running
}
```

---

## File Structure for Export

When exporting, organize files as:
```
mobile-ui/
├── screens/
│   ├── SplashScreen.jsx
│   ├── HomeScreen.jsx (Camera)
│   ├── PreviewScreen.jsx
│   ├── ResultScreen.jsx
│   ├── LibraryScreen.jsx
│   ├── ViewerScreen.jsx
│   └── SettingsScreen.jsx
├── components/
│   ├── GlassCard.jsx
│   ├── CaptureButton.jsx
│   ├── CornerOverlay.jsx
│   ├── DocumentCard.jsx
│   ├── ActionButton.jsx
│   └── BottomSheet.jsx
├── services/
│   └── api.js (API calls)
├── styles/
│   └── theme.js (colors, fonts)
├── navigation/
│   └── AppNavigator.jsx
└── App.jsx
```

---

## Important Notes

1. **Make it feel premium** - Every interaction should feel polished
2. **Dark theme first** - Optimize for OLED screens
3. **Smooth 60fps animations** - Use native driver where possible  
4. **Accessibility** - Include proper labels and contrast ratios
5. **Placeholder images** - Use gradient placeholders for thumbnails
6. **Loading states** - Every async action needs a loading state
7. **Error handling** - Friendly error messages with retry options

---

## Example Accent Interactions

- When document is detected: corners glow cyan → green with subtle haptic
- When saving: circular progress around save button
- When sharing: card lifts up with 3D tilt effect
- Pull to refresh: custom scanner beam animation
- Success: confetti particles burst from center

---

**End of Prompt - Paste this into Google Stitch to generate the UI!**
