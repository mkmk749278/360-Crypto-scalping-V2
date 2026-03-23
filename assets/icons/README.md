# 360 Crypto Eye — Channel Icons

Professionally designed SVG icons for all signal channels in the 360 Crypto Eye engine.
All icons are self-contained, scalable (512×512), and optimised for Telegram profile pictures and branding assets.

---

## Icon Inventory

| File | Channel | Emoji | Theme | Colors |
|------|---------|-------|-------|--------|
| `scalp.svg` | **360_SCALP** | ⚡ | High-Frequency Scalping | Gold + Cyan on dark `#0A0A1A` |
| `swing.svg` | **360_SWING** | 🏛️ | Institutional Swing | Gold + Royal Blue on dark navy `#0A0A2E` |
| `spot.svg` | **360_SPOT** | 📈 | Spot DCA Accumulation | Gold + Green on dark `#0A1A0A` |
| `gem.svg` | **360_GEM** | 💎 | Macro Reversal Scanner | Gold + Purple on dark `#1A0A2E` |
| `main_logo.svg` | **360 CRYPTO EYE** | 🔮 | Main Brand Logo | Gold + Electric Blue on black `#0A0A0A` |

---

## Design Language

All icons share a consistent visual identity:

- **Background:** Dark, near-black backgrounds for Telegram dark-mode compatibility
- **Accent:** Gold `#FFD700` as the universal brand accent across all icons
- **Glow effects:** SVG-native filters (`feGaussianBlur`) for neon glow and depth
- **Corner marks:** Thin gold bracket corners unify the series
- **Underline bar:** Coloured bar beneath the channel name in each icon's primary accent colour
- **Size:** 512×512 px — crop-safe for Telegram's circle crop at any resolution

---

## Usage

### As Telegram Channel Profile Pictures
1. Open each SVG in a browser or image editor (e.g. Inkscape, Figma, or Chrome)
2. Export / screenshot at **512×512 px** as PNG
3. Upload directly as the channel profile photo in Telegram

> Tip: Telegram automatically applies a circular crop, so the central element of each icon is designed to remain visible after cropping.

### As Web / App Assets
Reference directly in HTML or embed inline — all SVGs have no external dependencies:

```html
<img src="assets/icons/scalp.svg" alt="360 SCALP" width="64" height="64">
```

Or inline for CSS control:

```html
<svg><!-- contents of scalp.svg --></svg>
```

---

## Colour Palette

| Token | Hex | Usage |
|-------|-----|-------|
| Gold | `#FFD700` | Universal accent, text, key elements |
| Gold Light | `#FFF176` | Highlights, reflections |
| Gold Dark | `#B8860B` | Shadows, depth on gold elements |
| Cyan | `#00E5FF` | SCALP channel accent |
| Royal Blue | `#1A47B8` | SWING channel accent |
| Green | `#00C853` | SPOT channel accent |
| Purple | `#7C4DFF` | GEM channel accent |
| Electric Blue | `#00A3FF` | Main logo / brand accent |
