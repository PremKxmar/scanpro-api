export interface DocumentItem {
  id: string;
  title: string;
  date: string;
  size: string;
  thumbnail: string;
  pageCount?: number;
  type: 'pdf' | 'jpg';
}

export type Screen = 
  | 'splash'
  | 'onboarding'
  | 'library'
  | 'tools'
  | 'camera'
  | 'edit'
  | 'result'
  | 'viewer'
  | 'settings';

export interface ThemeContextType {
  isDark: boolean;
  toggleTheme: () => void;
}
