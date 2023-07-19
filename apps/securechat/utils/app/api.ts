import { Plugin, PluginID } from '@/types/plugin';

export const getEndpoint = (plugin: Plugin | null, url: string | null = null) => {

  if (url !== null) {
    return `${process.env.NEXT_PUBLIC_BASE_URL ? (process.env.NEXT_PUBLIC_BASE_URL + '/') : ''}api/${url}`;
  }

  if (!plugin) {
    return `${process.env.NEXT_PUBLIC_BASE_URL ? (process.env.NEXT_PUBLIC_BASE_URL + '/') : ''}api/chat`;
  }

  if (plugin.id === PluginID.GOOGLE_SEARCH) {
    return 'api/google';
  }

  return `${process.env.NEXT_PUBLIC_BASE_URL ? (process.env.NEXT_PUBLIC_BASE_URL + '/') : ''}api/chat`;
};
