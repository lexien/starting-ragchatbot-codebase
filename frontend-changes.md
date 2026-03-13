# Frontend Changes: Light/Dark Mode Toggle

## Summary
Implemented a full light/dark mode toggle with accessible sun/moon icon button (fixed, top-right), smooth CSS transitions, WCAG AA-compliant color palette for both themes, and `localStorage` persistence.

---

## Files Modified

### `frontend/style.css`

#### CSS Variables (`:root`)
Reorganized the variable block with named sections. Added new variables to replace all previously hardcoded colors:

| New variable | Purpose |
|---|---|
| `--source-pill-bg/border/color` | Source tag chips (default dark) |
| `--source-pill-hover-*` | Source tag hover state |
| `--error-bg/color/border` | Error status messages |
| `--success-bg/color/border` | Success status messages |
| `--code-bg` | Inline code and pre blocks |
| `--welcome-shadow` | Welcome message drop shadow |

Added `color-scheme: dark` so browser chrome (scrollbars, inputs) matches.

#### `[data-theme="light"]` block
Full light-theme override for every variable. All text/background contrast pairs verified against WCAG AA (≥4.5:1):

| Variable | Light value | Contrast on `--surface` (#fff) |
|---|---|---|
| `--text-primary` | `#0f172a` | ~16:1 ✓ AAA |
| `--text-secondary` | `#475569` | ~5.9:1 ✓ AA |
| `--source-pill-color` | `#1e40af` | ~8.6:1 ✓ AAA |
| `--error-color` | `#b91c1c` | ~5.9:1 ✓ AA |
| `--success-color` | `#15803d` | ~5.7:1 ✓ AA |

Added `color-scheme: light` for browser chrome.

#### Hardcoded colors converted to variables
- `.source-pill` — bg, border, color, hover states
- `.message-content code` and `pre` — background
- `.message.welcome-message .message-content` — box-shadow
- `.error-message` — bg, color, border
- `.success-message` — bg, color, border

#### Smooth theme transitions
Added a shared `transition: background-color 0.3s ease, color 0.3s ease, border-color 0.3s ease, box-shadow 0.3s ease` rule targeting all key surfaces: `.sidebar`, `.chat-messages`, `.chat-input-container`, `.message-content`, `#chatInput`, `.stat-item`, `.suggested-item`, `.source-pill`, `.error-message`, `.success-message`, `.theme-toggle`. Also added `transition` to `body`.

#### Theme Toggle button styles (`.theme-toggle`)
- `position: fixed; top: 1rem; right: 1rem` — top-right placement
- 40×40px circle, `border: 1px solid var(--border-color)`
- Hover: scale(1.1), primary border, focus ring shadow
- `:focus-visible` ring using `--focus-ring` (keyboard accessible)
- `.icon-sun` / `.icon-moon` positioned absolutely inside, animated with `opacity` + `rotate/scale` transforms (0.4s ease). Dark mode shows moon; light mode shows sun.

---

### `frontend/index.html`
- Added `<button class="theme-toggle" id="themeToggle">` immediately before `.container`
- Contains two inline SVGs: sun icon (`.icon-sun`) and moon icon (`.icon-moon`), both `aria-hidden="true"`
- Button has `aria-label="Toggle light/dark mode"` (updated dynamically by JS on load and on each toggle)

---

### `frontend/script.js`

#### Theme management (runs before `DOMContentLoaded`)
- **`initTheme()`** — reads `localStorage.getItem('theme')` (defaults to `"dark"`) and applies it via `document.documentElement.setAttribute('data-theme', ...)`. Runs immediately to prevent flash of wrong theme.
- **`toggleTheme()`** — flips `data-theme` between `"light"` / `"dark"`, persists to `localStorage`, updates button `aria-label` to describe the next action.

#### `DOMContentLoaded` handler
- Sets the correct initial `aria-label` on `#themeToggle` based on the theme already active (set by `initTheme()`).
- Attaches `click` listener calling `toggleTheme()`.
