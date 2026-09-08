"""Add daily market price history.

Revision ID: 0005_market_prices
Revises: 0004_observability
"""

from alembic import op
import sqlalchemy as sa


revision = "0005_market_prices"
down_revision = "0004_observability"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Some early development builds used SQLAlchemy create_all before Alembic.
    # Adopt that identical table safely, then let Alembic be the sole owner.
    inspector = sa.inspect(op.get_bind())
    if "market_prices" not in inspector.get_table_names():
        op.create_table(
            "market_prices",
            sa.Column("id", sa.Integer(), nullable=False),
            sa.Column("company_id", sa.Integer(), nullable=False),
            sa.Column("trade_date", sa.DateTime(), nullable=False),
            sa.Column("open", sa.Float(), nullable=True),
            sa.Column("close", sa.Float(), nullable=True),
            sa.Column("high", sa.Float(), nullable=True),
            sa.Column("low", sa.Float(), nullable=True),
            sa.Column("volume", sa.Float(), nullable=True),
            sa.Column("amount", sa.Float(), nullable=True),
            sa.Column("change_pct", sa.Float(), nullable=True),
            sa.Column("source_name", sa.String(length=64), nullable=False, server_default="东方财富"),
            sa.Column("created_at", sa.DateTime(), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=False),
            sa.Column("updated_at", sa.DateTime(), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=False),
            sa.ForeignKeyConstraint(["company_id"], ["companies.id"]),
            sa.PrimaryKeyConstraint("id"),
            sa.UniqueConstraint("company_id", "trade_date", name="uq_market_price_company_date"),
        )
        op.create_index("ix_market_prices_company_id", "market_prices", ["company_id"], unique=False)
        op.create_index("ix_market_prices_trade_date", "market_prices", ["trade_date"], unique=False)
        op.create_index("ix_market_prices_company_date", "market_prices", ["company_id", "trade_date"], unique=False)


def downgrade() -> None:
    op.drop_index("ix_market_prices_company_date", table_name="market_prices")
    op.drop_index("ix_market_prices_trade_date", table_name="market_prices")
    op.drop_index("ix_market_prices_company_id", table_name="market_prices")
    op.drop_table("market_prices")
