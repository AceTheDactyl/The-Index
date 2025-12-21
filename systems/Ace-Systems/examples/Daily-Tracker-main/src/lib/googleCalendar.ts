// Google Calendar API Configuration
const SCOPES = 'https://www.googleapis.com/auth/calendar.events';
const DISCOVERY_DOC = 'https://www.googleapis.com/discovery/v1/apis/calendar/v3/rest';
const TOKEN_STORAGE_KEY = 'pulse-gcal-token';

interface GoogleCalendarConfig {
  apiKey: string;
  clientId: string;
  calendarId: string;
}

interface StoredToken {
  access_token: string;
  expires_at: number;
  token_type: string;
  scope: string;
}

declare global {
  interface Window {
    gapi: any;
    google: any;
  }
}

export class GoogleCalendarService {
  private tokenClient: any;
  private config: GoogleCalendarConfig;
  private onAuthChange?: (isAuthenticated: boolean) => void;

  constructor(config: GoogleCalendarConfig) {
    this.config = config;
  }

  /**
   * Set callback for auth state changes
   */
  setAuthChangeCallback(callback: (isAuthenticated: boolean) => void) {
    this.onAuthChange = callback;
  }

  async initialize() {
    await this.loadGoogleScripts();
    await this.initializeGapi();
    this.initializeGis();

    // Try to restore saved token
    await this.restoreToken();
  }

  private loadGoogleScripts(): Promise<void> {
    return new Promise((resolve) => {
      let gapiLoaded = false;
      let gisLoaded = false;

      const checkBothLoaded = () => {
        if (gapiLoaded && gisLoaded) {
          resolve();
        }
      };

      // Load Google API script
      const gapiScript = document.createElement('script');
      gapiScript.src = 'https://apis.google.com/js/api.js';
      gapiScript.onload = () => {
        window.gapi.load('client', async () => {
          gapiLoaded = true;
          checkBothLoaded();
        });
      };
      document.body.appendChild(gapiScript);

      // Load Google Identity Services script
      const gisScript = document.createElement('script');
      gisScript.src = 'https://accounts.google.com/gsi/client';
      gisScript.onload = () => {
        gisLoaded = true;
        checkBothLoaded();
      };
      document.body.appendChild(gisScript);
    });
  }

  private async initializeGapi() {
    await window.gapi.client.init({
      apiKey: this.config.apiKey,
      discoveryDocs: [DISCOVERY_DOC],
    });
  }

  private initializeGis() {
    this.tokenClient = window.google.accounts.oauth2.initTokenClient({
      client_id: this.config.clientId,
      scope: SCOPES,
      callback: '', // defined later
    });
  }

  /**
   * Save token to localStorage
   */
  private saveToken(token: any) {
    try {
      const storedToken: StoredToken = {
        access_token: token.access_token,
        expires_at: Date.now() + (token.expires_in * 1000),
        token_type: token.token_type,
        scope: token.scope
      };
      localStorage.setItem(TOKEN_STORAGE_KEY, JSON.stringify(storedToken));
      console.log('Google Calendar token saved');
    } catch (e) {
      console.error('Failed to save token:', e);
    }
  }

  /**
   * Restore token from localStorage
   */
  private async restoreToken(): Promise<boolean> {
    try {
      const stored = localStorage.getItem(TOKEN_STORAGE_KEY);
      if (!stored) return false;

      const token: StoredToken = JSON.parse(stored);

      // Check if token is expired (with 5 min buffer)
      if (token.expires_at < Date.now() + 300000) {
        console.log('Stored token expired, clearing');
        localStorage.removeItem(TOKEN_STORAGE_KEY);
        return false;
      }

      // Restore token to gapi client
      window.gapi.client.setToken({
        access_token: token.access_token,
        token_type: token.token_type,
        scope: token.scope,
        expires_in: Math.floor((token.expires_at - Date.now()) / 1000)
      });

      console.log('Google Calendar token restored from storage');
      this.onAuthChange?.(true);
      return true;
    } catch (e) {
      console.error('Failed to restore token:', e);
      localStorage.removeItem(TOKEN_STORAGE_KEY);
      return false;
    }
  }

  /**
   * Check if currently authenticated
   */
  isAuthenticated(): boolean {
    const token = window.gapi?.client?.getToken();
    return token !== null && token !== undefined;
  }

  /**
   * Check if there's a stored token that might be valid
   */
  hasStoredToken(): boolean {
    try {
      const stored = localStorage.getItem(TOKEN_STORAGE_KEY);
      if (!stored) return false;
      const token: StoredToken = JSON.parse(stored);
      return token.expires_at > Date.now() + 300000;
    } catch {
      return false;
    }
  }

  async authenticate(): Promise<boolean> {
    return new Promise((resolve) => {
      this.tokenClient.callback = async (resp: any) => {
        if (resp.error !== undefined) {
          this.onAuthChange?.(false);
          resolve(false);
          return;
        }

        // Save the token for persistence
        const token = window.gapi.client.getToken();
        if (token) {
          this.saveToken(token);
        }

        this.onAuthChange?.(true);
        resolve(true);
      };

      if (window.gapi.client.getToken() === null) {
        this.tokenClient.requestAccessToken({ prompt: 'consent' });
      } else {
        this.tokenClient.requestAccessToken({ prompt: '' });
      }
    });
  }

  /**
   * Sign out and clear stored token
   */
  signOut() {
    const token = window.gapi.client.getToken();
    if (token) {
      window.google.accounts.oauth2.revoke(token.access_token);
      window.gapi.client.setToken(null);
    }
    localStorage.removeItem(TOKEN_STORAGE_KEY);
    this.onAuthChange?.(false);
    console.log('Signed out of Google Calendar');
  }

  async createEvent(event: {
    summary: string;
    description?: string;
    start: string; // ISO datetime
    end: string; // ISO datetime
    colorId?: string;
  }) {
    const response = await window.gapi.client.calendar.events.insert({
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
    const response = await window.gapi.client.calendar.events.list({
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
    await window.gapi.client.calendar.events.delete({
      calendarId: this.config.calendarId,
      eventId: eventId,
    });
  }

  async updateEvent(eventId: string, updates: any) {
    const response = await window.gapi.client.calendar.events.patch({
      calendarId: this.config.calendarId,
      eventId: eventId,
      resource: updates,
    });
    return response.result;
  }
}
