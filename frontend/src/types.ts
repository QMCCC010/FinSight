export interface Company {
  id: number
  stock_code: string
  name: string
  full_name?: string
  exchange?: string
  industry?: string
  tracking_mode: string
  last_crawled_at?: string
}

export interface DocumentItem {
  id: number
  company_id: number
  document_type: string
  title: string
  source_name: string
  source_url: string
  published_at?: string
  summary?: string
  status: string
}

export interface ReportCitation {
  number: number
  document_id: number
  title: string
  source_type: string
  source_name: string
  source_url: string
  published_at?: string
  page?: number
  quote: string
  credibility: string
}

export interface ReportItem {
  id: number
  company_id: number
  report_type: string
  title: string
  status: string
  progress: number
  status_text?: string
  error?: string
  source_types: string[]
  selected_document_ids: number[]
  content_markdown: string
  citations: ReportCitation[]
  completed_at?: string
  created_at: string
  updated_at: string
}

export interface ReportMaterialDocument extends DocumentItem {
  usable: boolean
}

export interface ReportMaterial {
  company: { id: number; name: string; stock_code: string }
  total: number
  usable: number
  last_crawled_at?: string
  last_indexed_at?: string
  is_collecting: boolean
  freshness: Record<string, { label: string; count: number; latest_at?: string; is_fresh: boolean }>
  warnings: string[]
  documents: ReportMaterialDocument[]
}

export interface ReportVersion {
  id: number
  report_id: number
  version_number: number
  title: string
  content_markdown: string
  citations: ReportCitation[]
  change_type: string
  created_at: string
}
