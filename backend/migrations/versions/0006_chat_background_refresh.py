"""Track non-blocking knowledge refreshes for chat answers.

Revision ID: 0006_chat_background_refresh
Revises: 0005_market_prices
"""

from alembic import op
import sqlalchemy as sa


revision = "0006_chat_background_refresh"
down_revision = "0005_market_prices"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("chat_messages", sa.Column("refresh_run_id", sa.Integer(), nullable=True))
    op.add_column("chat_messages", sa.Column("refresh_status", sa.String(length=20), nullable=True))
    op.add_column("chat_messages", sa.Column("refresh_status_text", sa.String(length=255), nullable=True))
    op.add_column("chat_messages", sa.Column("refresh_requested_at", sa.DateTime(), nullable=True))
    op.add_column("chat_messages", sa.Column("refresh_completed_at", sa.DateTime(), nullable=True))
    op.create_index("ix_chat_messages_refresh_run_id", "chat_messages", ["refresh_run_id"], unique=False)
    op.create_index("ix_chat_messages_refresh_status", "chat_messages", ["refresh_status"], unique=False)
    op.create_foreign_key(
        "fk_chat_messages_refresh_run_id_crawl_runs",
        "chat_messages",
        "crawl_runs",
        ["refresh_run_id"],
        ["id"],
    )


def downgrade() -> None:
    op.drop_constraint("fk_chat_messages_refresh_run_id_crawl_runs", "chat_messages", type_="foreignkey")
    op.drop_index("ix_chat_messages_refresh_status", table_name="chat_messages")
    op.drop_index("ix_chat_messages_refresh_run_id", table_name="chat_messages")
    op.drop_column("chat_messages", "refresh_completed_at")
    op.drop_column("chat_messages", "refresh_requested_at")
    op.drop_column("chat_messages", "refresh_status_text")
    op.drop_column("chat_messages", "refresh_status")
    op.drop_column("chat_messages", "refresh_run_id")
