from app.collector.adapters.announcement import AnnouncementAdapter
from app.collector.adapters.news import NewsAdapter
from app.collector.adapters.research import ResearchReportAdapter
from app.collector.adapters.social import SocialAdapter

ADAPTERS = {
    "RESEARCH_REPORT": ResearchReportAdapter,
    "NEWS": NewsAdapter,
    "ANNOUNCEMENT": AnnouncementAdapter,
    "SOCIAL": SocialAdapter,
}

__all__ = ["ADAPTERS"]

