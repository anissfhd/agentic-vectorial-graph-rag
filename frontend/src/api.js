import axios from 'axios'

const api = axios.create({
  baseURL: import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000',
  timeout: 120000,
})

export const getJson = async (path) => (await api.get(path)).data
export const postJson = async (path, payload = {}) => (await api.post(path, payload)).data

export default api

