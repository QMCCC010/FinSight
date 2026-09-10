"""Add persistent idempotency keys for chat submissions.

Revision ID: 0011_chat_request_idempotency
Revises: 0010_document_parse_quality
"""

from alembic import op
import sqlalchemy as sa


revision = "0011_chat_request_idempotency"
down_revision = "0010_document_parse_quality"
branch_labels = None
depends_on = None


CONSTRAINT_NAME = "uq_chat_message_client_request"
INDEX_NAME = "ix_chat_messages_client_request_id"


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    columns = {item["name"] for item in inspector.get_columns("chat_messages")}
    if "client_request_id" not in columns:
        op.add_column("chat_messages", sa.Column("client_request_id", sa.String(length=64), nullable=True))

    inspector = sa.inspect(bind)
    indexes = {item["name"] for item in inspector.get_indexes("chat_messages")}
    if INDEX_NAME not in indexes:
        op.create_index(INDEX_NAME, "chat_messages", ["client_request_id"], unique=False)

    constraints = {item["name"] for item in inspector.get_unique_constraints("chat_messages")}
    if CONSTRAINT_NAME not in constraints:
        op.create_unique_constraint(
            CONSTRAINT_NAME,
            "chat_messages",
            ["user_id", "session_id", "client_request_id"],
        )


def downgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    constraints = {item["name"] for item in inspector.get_unique_constraints("chat_messages")}
    if CONSTRAINT_NAME in constraints:
        op.drop_constraint(CONSTRAINT_NAME, "chat_messages", type_="unique")
    indexes = {item["name"] for item in inspector.get_indexes("chat_messages")}
    if INDEX_NAME in indexes:
        op.drop_index(INDEX_NAME, table_name="chat_messages")
    columns = {item["name"] for item in inspector.get_columns("chat_messages")}
    if "client_request_id" in columns:
        op.drop_column("chat_messages", "client_request_id")
