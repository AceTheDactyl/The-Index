# Pulse Check Rhythm

A biologically-informed daily rhythm tracker that maps your energy waves, schedules anchored activities, and maintains flow state through personalized task management.

## 🎯 Project Goals

1. **Deploy to GitHub Pages** - Automated CI/CD pipeline
2. **Google Calendar Integration** - Bidirectional sync with Google Calendar
3. **Persistent Storage** - Browser localStorage for offline-first functionality

---

## 🤖 Instructions for Claude Code

This README is designed to guide autonomous development. Follow these sections in order.

---

## Phase 1: Initial Setup & GitHub Pages Deployment

### Step 1: Initialize Vite React TypeScript Project

```bash
npm create vite@latest pulse-check-rhythm -- --template react-ts
cd pulse-check-rhythm
npm install
npm install lucide-react
```

### Step 2: Replace App.tsx

- Copy the entire TSX artifact code into `src/App.tsx`
- Remove any imports/exports that conflict with the Vite setup
- Ensure the default export is `export default function App()`

### Step 3: Update Storage Functions

Replace the storage shim with localStorage-only implementation:

```typescript
const storageGet = async (key: string): Promise<string | null> => {
  try {
    return localStorage.getItem(key);
  } catch (error) {
    console.error('Storage get error:', error);
    return null;
  }
};

const storageSet = async (key: string, value: string) => {
  try {
    localStorage.setItem(key, value);
  } catch (error) {
    console.error('Storage set error:', error);
  }
};
```

### Step 4: Configure Vite for GitHub Pages

Update `vite.config.ts`:

```typescript
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  base: '/pulse-check-rhythm/',
  build: {
    outDir: 'dist',
    sourcemap: true
  }
})
```

### Step 5: Add GitHub Actions Workflow

Create `.github/workflows/deploy.yml`:

```yaml
name: Deploy to GitHub Pages

on:
  push:
    branches: [ main ]
  workflow_dispatch:

permissions:
  contents: read
  pages: write
  id-token: write

concurrency:
  group: "pages"
  cancel-in-progress: false

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout
        uses: actions/checkout@v4
      
      - name: Setup Node
        uses: actions/setup-node@v4
        with:
          node-version: '20'
          cache: 'npm'
      
      - name: Install dependencies
        run: npm ci
      
      - name: Build
        run: npm run build
      
      - name: Setup Pages
        uses: actions/configure-pages@v4
      
      - name: Upload artifact
        uses: actions/upload-pages-artifact@v3
        with:
          path: './dist'
  
  deploy:
    environment:
      name: github-pages
      url: ${{ steps.deployment.outputs.page_url }}
    runs-on: ubuntu-latest
    needs: build
    steps:
      - name: Deploy to GitHub Pages
        id: deployment
        uses: actions/deploy-pages@v4
```

### Step 6: Update package.json

Add build and preview scripts:

```json
{
  "scripts": {
    "dev": "vite",
    "build": "tsc && vite build",
    "preview": "vite preview",
    "lint": "eslint . --ext ts,tsx --report-unused-disable-directives --max-warnings 0"
  }
}
```

### Step 7: Enable GitHub Pages

1. Push code to GitHub repository
2. Go to repository Settings → Pages
3. Source: "GitHub Actions"
4. Wait for workflow to complete
5. Site will be live at `https://[username].github.io/pulse-check-rhythm/`

---

## Phase 2: Google Calendar Integration

### Architecture Overview

The integration provides:
- **Export beats to Google Calendar** ✅ - Create calendar events from scheduled check-ins
- **Import calendar events** ✅ - Import Google Calendar events as beats in the daily view
- **Two-way sync** ✅ - Auto-sync when enabled, manual import on demand
- **Wave tagging** ✅ - Color-code events by wave type and auto-assign waves based on time

### Features Implemented

1. **Connect Google Calendar** - One-click OAuth authentication
2. **Auto-sync to Calendar** - Toggle to automatically export beats to Google Calendar
3. **Import Events** - Button to import today's Google Calendar events as beats
4. **Wave Assignment** - Imported events automatically assigned to appropriate wave based on time
5. **Duplicate Prevention** - Smart filtering prevents importing the same event twice

### Step 1: Setup Google Calendar API

Create `src/lib/googleCalendar.ts`:

```typescript
// Google Calendar API Configuration
const SCOPES = 'https://www.googleapis.com/auth/calendar.events';
const DISCOVERY_DOC = 'https://www.googleapis.com/discovery/v1/apis/calendar/v3/rest';

interface GoogleCalendarConfig {
  apiKey: string;
  clientId: string;
  calendarId: string;
}

export class GoogleCalendarService {
  private tokenClient: any;
  private gapiInited = false;
  private gisInited = false;

  constructor(private config: GoogleCalendarConfig) {}

  async initialize() {
    await this.loadGoogleScripts();
    await this.initializeGapi();
    this.initializeGis();
  }

  private loadGoogleScripts(): Promise<void> {
    return new Promise((resolve) => {
      // Load Google API script
      const gapiScript = document.createElement('script');
      gapiScript.src = 'https://apis.google.com/js/api.js';
      gapiScript.onload = () => {
        // @ts-ignore
        gapi.load('client', async () => {
          this.gapiInited = true;
          if (this.gisInited) resolve();
        });
      };
      document.body.appendChild(gapiScript);

      // Load Google Identity Services script
      const gisScript = document.createElement('script');
      gisScript.src = 'https://accounts.google.com/gsi/client';
      gisScript.onload = () => {
        this.gisInited = true;
        if (this.gapiInited) resolve();
      };
      document.body.appendChild(gisScript);
    });
  }

  private async initializeGapi() {
    // @ts-ignore
    await gapi.client.init({
      apiKey: this.config.apiKey,
      discoveryDocs: [DISCOVERY_DOC],
    });
  }

  private initializeGis() {
    // @ts-ignore
    this.tokenClient = google.accounts.oauth2.initTokenClient({
      client_id: this.config.clientId,
      scope: SCOPES,
      callback: '', // defined later
    });
  }

  async authenticate(): Promise<boolean> {
    return new Promise((resolve) => {
      this.tokenClient.callback = async (resp: any) => {
        if (resp.error !== undefined) {
          resolve(false);
          return;
        }
        resolve(true);
      };

      // @ts-ignore
      if (gapi.client.getToken() === null) {
        this.tokenClient.requestAccessToken({ prompt: 'consent' });
      } else {
        this.tokenClient.requestAccessToken({ prompt: '' });
      }
    });
  }

  async createEvent(event: {
    summary: string;
    description?: string;
    start: string; // ISO datetime
    end: string; // ISO datetime
    colorId?: string;
  }) {
    // @ts-ignore
    const response = await gapi.client.calendar.events.insert({
      calendarId: this.config.calendarId,
      resource: {
        summary: event.summary,
        description: event.description,
        start: {
          dateTime: event.start,
          timeZone: Intl.DateTimeFormat().resolvedOptions().timeZone,
        },
        end: {
          dateTime: event.end,
          timeZone: Intl.DateTimeFormat().resolvedOptions().timeZone,
        },
        colorId: event.colorId || '1',
      },
    });
    return response.result;
  }

  async listEvents(timeMin: string, timeMax: string) {
    // @ts-ignore
    const response = await gapi.client.calendar.events.list({
      calendarId: this.config.calendarId,
      timeMin: timeMin,
      timeMax: timeMax,
      showDeleted: false,
      singleEvents: true,
      orderBy: 'startTime',
    });
    return response.result.items || [];
  }

  async deleteEvent(eventId: string) {
    // @ts-ignore
    await gapi.client.calendar.events.delete({
      calendarId: this.config.calendarId,
      eventId: eventId,
    });
  }

  async updateEvent(eventId: string, updates: any) {
    // @ts-ignore
    const response = await gapi.client.calendar.events.patch({
      calendarId: this.config.calendarId,
      eventId: eventId,
      resource: updates,
    });
    return response.result;
  }
}
```

### Step 2: Environment Configuration

Create `.env.local`:

```
VITE_GOOGLE_CLIENT_ID=your_client_id_here.apps.googleusercontent.com
VITE_GOOGLE_API_KEY=your_api_key_here
VITE_GOOGLE_CALENDAR_ID=primary
```

**Important**: Add `.env.local` to `.gitignore`

### Step 3: Setup Google Cloud Project

Instructions to add to README for manual configuration:

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create new project: "Pulse Check Rhythm"
3. Enable Google Calendar API
4. Create OAuth 2.0 Client ID (Web application)
   - Authorized JavaScript origins: 
     - `http://localhost:5173`
     - `https://[username].github.io`
   - Authorized redirect URIs:
     - `http://localhost:5173`
     - `https://[username].github.io/pulse-check-rhythm/`
5. Create API Key (restrict to Calendar API)
6. Add to `.env.local`

### Step 4: Integrate into App.tsx

Add Google Calendar hooks and state management:

```typescript
// Add to imports
import { GoogleCalendarService } from './lib/googleCalendar';

// Add to App component state
const [gcalService, setGcalService] = useState<GoogleCalendarService | null>(null);
const [gcalAuthed, setGcalAuthed] = useState(false);
const [syncEnabled, setSyncEnabled] = useState(false);

// Initialize Google Calendar
useEffect(() => {
  const initGCal = async () => {
    const service = new GoogleCalendarService({
      apiKey: import.meta.env.VITE_GOOGLE_API_KEY,
      clientId: import.meta.env.VITE_GOOGLE_CLIENT_ID,
      calendarId: import.meta.env.VITE_GOOGLE_CALENDAR_ID || 'primary',
    });
    
    try {
      await service.initialize();
      setGcalService(service);
    } catch (error) {
      console.error('Failed to initialize Google Calendar:', error);
    }
  };
  
  initGCal();
}, []);

// Sync function
const syncToGoogleCalendar = async (checkIn: CheckIn) => {
  if (!gcalService || !gcalAuthed || !syncEnabled) return;
  
  try {
    const start = new Date(checkIn.slot);
    const end = new Date(start.getTime() + 30 * 60000); // 30 min default
    
    await gcalService.createEvent({
      summary: `${checkIn.category}: ${checkIn.task}`,
      description: checkIn.note || '',
      start: start.toISOString(),
      end: end.toISOString(),
      colorId: getGoogleCalendarColor(checkIn.waveId),
    });
  } catch (error) {
    console.error('Failed to sync to Google Calendar:', error);
  }
};

const getGoogleCalendarColor = (waveId?: string): string => {
  // Google Calendar color IDs
  const colorMap: Record<string, string> = {
    'cyan': '7',    // Cyan
    'purple': '3',  // Purple
    'blue': '9',    // Blue
    'orange': '6',  // Orange
  };
  
  const color = getWaveColor(waveId);
  return colorMap[color] || '1';
};
```

### Step 5: Add UI Controls

Add Google Calendar authentication button and sync toggle to the header section:

```typescript
{/* Google Calendar Integration */}
<div className="flex items-center gap-2">
  {gcalService && (
    <>
      {!gcalAuthed ? (
        <button
          onClick={async () => {
            const authed = await gcalService.authenticate();
            setGcalAuthed(authed);
          }}
          className="px-3 py-1.5 rounded-lg bg-green-600 hover:bg-green-500 text-sm flex items-center gap-2"
        >
          <Calendar className="w-4 h-4" />
          Connect Google Calendar
        </button>
      ) : (
        <label className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-gray-900/70 border border-gray-800 text-sm cursor-pointer">
          <input
            type="checkbox"
            checked={syncEnabled}
            onChange={(e) => setSyncEnabled(e.target.checked)}
            className="rounded"
          />
          Sync to Calendar
        </label>
      )}
    </>
  )}
</div>
```

### Step 6: Auto-sync Logic

Modify `scheduleBeat` to auto-sync when enabled:

```typescript
const scheduleBeat = async (
  category: string,
  task: string,
  when: Date,
  note?: string,
  waveId?: string,
  isAnchor?: boolean
) => {
  const entry: CheckIn = {
    id: Date.now().toString() + Math.random(),
    category,
    task,
    waveId,
    slot: when.toISOString(),
    loggedAt: new Date().toISOString(),
    note,
    done: false,
    isAnchor,
  };
  
  setCheckIns(prev => [entry, ...prev]);
  
  // Auto-sync to Google Calendar
  if (syncEnabled && gcalAuthed) {
    await syncToGoogleCalendar(entry);
  }
};
```

---

## Phase 3: Testing & Validation

### Local Development Testing

```bash
npm run dev
```

Verify:
- [ ] All components render correctly
- [ ] localStorage persistence works
- [ ] Wave setup wizard functions
- [ ] Daily anchors can be set
- [ ] Check-ins can be scheduled
- [ ] Journal entries save
- [ ] Google Calendar auth flow works
- [ ] Events sync to Google Calendar

### Production Build Testing

```bash
npm run build
npm run preview
```

Verify:
- [ ] Build completes without errors
- [ ] Preview shows production bundle working
- [ ] All features function in production mode

### GitHub Pages Deployment Testing

After pushing to main:
- [ ] GitHub Actions workflow completes successfully
- [ ] Site loads at GitHub Pages URL
- [ ] All features work on deployed site
- [ ] Google Calendar OAuth works with production URL

---

## 📁 Expected Project Structure

```
pulse-check-rhythm/
├── .github/
│   └── workflows/
│       └── deploy.yml
├── src/
│   ├── lib/
│   │   └── googleCalendar.ts
│   ├── App.tsx
│   ├── main.tsx
│   └── index.css
├── .env.local (gitignored)
├── .gitignore
├── index.html
├── package.json
├── tsconfig.json
├── vite.config.ts
└── README.md
```

---

## 🔧 Troubleshooting

### Common Issues

**GitHub Pages 404 Error**
- Verify `base` in `vite.config.ts` matches repository name
- Check GitHub Pages source is set to "GitHub Actions"

**Google Calendar API Errors**
- Verify API is enabled in Google Cloud Console
- Check OAuth consent screen is configured
- Ensure authorized origins match deployment URL
- Verify API key restrictions allow Calendar API

**localStorage Not Persisting**
- Check browser privacy settings
- Verify no ad blockers interfering
- Test in incognito/private mode

**Build Failures**
- Run `npm ci` to ensure clean dependencies
- Check TypeScript errors with `npm run build`
- Verify all imports are correct

---

## 🚀 Deployment Checklist

- [ ] Code pushed to GitHub repository
- [ ] GitHub Pages enabled with Actions source
- [ ] `.env.local` created with Google credentials
- [ ] Google Cloud OAuth configured with production URL
- [ ] GitHub Actions workflow completed successfully
- [ ] Site accessible at GitHub Pages URL
- [ ] Google Calendar authentication working
- [ ] Event sync functioning bidirectionally
- [ ] All features tested in production

---

## 📝 Future Enhancements

Potential features for iteration:
- Recurring anchor templates
- Wave statistics and analytics
- Export rhythm data to CSV/JSON
- Dark/light theme toggle
- Mobile PWA installation
- Notifications for upcoming beats
- Multi-calendar support
- Share rhythm templates with others
- Real-time bidirectional sync (webhook-based)

---

## 🤝 Contributing

This project is designed for personal use and autonomous development via Claude Code. Feel free to fork and customize for your own rhythm needs.

---

## 📄 License

MIT License - Use freely, modify as needed, make it your own.

---

**Built with**: React + TypeScript + Vite + Tailwind CSS + Google Calendar API  
**Deployed on**: GitHub Pages  
**Designed for**: Neurodivergent-friendly rhythm tracking and flow state maintenance