"""Canonical concepts and the XBRL tags that can supply them.

Filers pick different us-gaap elements for the same economic line item and
switch elements over time (ASC 606 moved most companies from ``SalesRevenueNet``
to ``RevenueFromContractWithCustomerExcludingAssessedTax``). Each concept lists
its aliases in priority order. Two selection strategies exist:

* ``priority`` - the first alias with a value wins. Used where aliases are
  genuinely different measures (net income vs. net income including
  non-controlling interests) and the order expresses a preference.
* ``largest``  - the alias with the largest magnitude wins, priority breaking
  ties. Used for totals whose aliases are also filed for partial amounts: one
  company tags a small "cost of services" line with the element another uses
  for its entire cost of revenue. A total is never smaller than its parts, so
  the largest value is the consolidated line.

Tags that appear in filings but are absent from this dictionary are written to
``unmapped_tags`` so gaps are visible.
"""

from __future__ import annotations

from dataclasses import dataclass

DURATION = "duration"   # income statement / cash flow: measured over a span
INSTANT = "instant"     # balance sheet: measured at a point in time


@dataclass(frozen=True)
class Concept:
    key: str
    label: str
    period_type: str
    tags: tuple[str, ...]
    pick: str = "priority"
    unit: str = "USD"
    statement: str = ""     # "IS", "BS" or "CF"; inferred from period_type when blank

    def __post_init__(self):
        if not self.statement:
            object.__setattr__(self, "statement", "BS" if self.period_type == INSTANT else "IS")


CONCEPTS: tuple[Concept, ...] = (
    # ---- income statement -------------------------------------------------
    Concept("revenue", "Revenue", DURATION, (
        "Revenues",
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "RevenueFromContractWithCustomerIncludingAssessedTax",
        "SalesRevenueNet",
        "RevenuesNetOfInterestExpense",
        "SalesRevenueGoodsNet",
        "SalesRevenueServicesNet",
        "RegulatedAndUnregulatedOperatingRevenue",
        "ElectricUtilityRevenue",
        "OilAndGasRevenue",
        "RealEstateRevenueNet",
        "HealthCareOrganizationRevenue",
        "RevenuesExcludingInterestAndDividends",
        "OperatingLeaseLeaseIncome",                      # REITs: rental income is the revenue line
        "OperatingLeasesIncomeStatementLeaseRevenue",
    ), pick="largest"),
    Concept("cogs", "Cost of revenue", DURATION, (
        "CostOfGoodsAndServicesSold",
        "CostOfRevenue",
        "CostOfGoodsSold",
        "CostOfServices",
        "CostOfGoodsAndServiceExcludingDepreciationDepletionAndAmortization",
        "CostOfGoodsSoldExcludingDepreciationDepletionAndAmortization",
    ), pick="largest"),
    Concept("gross_profit", "Gross profit", DURATION, (
        "GrossProfit",
    )),
    Concept("operating_income", "Operating income", DURATION, (
        "OperatingIncomeLoss",
    )),
    Concept("pretax_income", "Pre-tax income", DURATION, (
        "IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest",
        "IncomeLossFromContinuingOperationsBeforeIncomeTaxesMinorityInterestAndIncomeLossFromEquityMethodInvestments",
        "IncomeLossFromContinuingOperationsBeforeIncomeTaxesDomestic",
    )),
    Concept("net_income", "Net income", DURATION, (
        "NetIncomeLoss",
        "ProfitLoss",
        "NetIncomeLossAvailableToCommonStockholdersBasic",
        "IncomeLossFromContinuingOperations",
    )),
    Concept("interest_expense", "Interest expense", DURATION, (
        "InterestExpense",
        "InterestExpenseNonoperating",
        "InterestExpenseDebt",
        "InterestAndDebtExpense",
        "InterestExpenseBorrowings",
        "InterestExpenseLongTermDebt",
    ), pick="largest"),
    Concept("depreciation_amortization", "Depreciation & amortization", DURATION, (
        "DepreciationDepletionAndAmortization",
        "DepreciationAndAmortization",
        "DepreciationAmortizationAndAccretionNet",
        "Depreciation",
        "DepreciationNonproduction",
    ), pick="largest"),
    # ---- balance sheet ----------------------------------------------------
    Concept("total_assets", "Total assets", INSTANT, (
        "Assets",
    )),
    Concept("current_assets", "Current assets", INSTANT, (
        "AssetsCurrent",
    )),
    Concept("cash", "Cash & equivalents", INSTANT, (
        "CashAndCashEquivalentsAtCarryingValue",
        "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents",
        "Cash",
    )),
    Concept("short_term_investments", "Short-term investments", INSTANT, (
        "ShortTermInvestments",
        "MarketableSecuritiesCurrent",
        "AvailableForSaleSecuritiesDebtSecuritiesCurrent",
        "AvailableForSaleSecuritiesCurrent",
    )),
    Concept("receivables", "Accounts receivable", INSTANT, (
        "AccountsReceivableNetCurrent",
        "ReceivablesNetCurrent",
        "AccountsNotesAndLoansReceivableNetCurrent",
        "AccountsAndOtherReceivablesNetCurrent",
    )),
    Concept("inventory", "Inventory", INSTANT, (
        "InventoryNet",
        "InventoryFinishedGoodsNetOfReserves",
    )),
    Concept("current_liabilities", "Current liabilities", INSTANT, (
        "LiabilitiesCurrent",
    )),
    Concept("accounts_payable", "Accounts payable", INSTANT, (
        "AccountsPayableCurrent",
        "AccountsPayableTradeCurrent",
        "AccountsPayableAndAccruedLiabilitiesCurrent",
    )),
    Concept("debt_total_reported", "Total debt (as reported)", INSTANT, (
        "DebtAndCapitalLeaseObligations",
        "DebtLongtermAndShorttermCombinedAmount",
        "LongTermDebt",
        "LongTermDebtAndCapitalLeaseObligationsIncludingCurrentMaturities",
    ), pick="largest"),
    Concept("debt_noncurrent", "Long-term debt, non-current", INSTANT, (
        "LongTermDebtNoncurrent",
        "LongTermDebtAndCapitalLeaseObligations",
        "LongTermNotesPayable",
        "SeniorNotes",
    ), pick="largest"),
    Concept("debt_current", "Debt, current", INSTANT, (
        "DebtCurrent",
        "LongTermDebtCurrent",
        "LongTermDebtAndCapitalLeaseObligationsCurrent",
        "ShortTermBorrowings",
        "CommercialPaper",
    ), pick="largest"),
    Concept("equity", "Shareholders' equity", INSTANT, (
        "StockholdersEquity",
        "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
    )),
    Concept("ppe_net", "Property, plant & equipment, net", INSTANT, (
        "PropertyPlantAndEquipmentNet",
        "PropertyPlantAndEquipmentAndFinanceLeaseRightOfUseAssetAfterAccumulatedDepreciationAndAmortization",
    )),
    Concept("goodwill", "Goodwill", INSTANT, (
        "Goodwill",
    )),
    Concept("intangibles", "Intangible assets, net", INSTANT, (
        "IntangibleAssetsNetExcludingGoodwill",
        "FiniteLivedIntangibleAssetsNet",
    ), pick="largest"),
    Concept("total_liabilities", "Total liabilities", INSTANT, (
        "Liabilities",
    )),
    Concept("retained_earnings", "Retained earnings", INSTANT, (
        "RetainedEarningsAccumulatedDeficit",
    )),
    Concept("deferred_revenue", "Deferred revenue, current", INSTANT, (
        "ContractWithCustomerLiabilityCurrent",
        "DeferredRevenueCurrent",
    )),
    # ---- more income statement ---------------------------------------------
    Concept("sga", "SG&A", DURATION, (
        "SellingGeneralAndAdministrativeExpense",
        "GeneralAndAdministrativeExpense",
    ), pick="largest"),
    Concept("rnd", "Research & development", DURATION, (
        "ResearchAndDevelopmentExpense",
        "ResearchAndDevelopmentExpenseExcludingAcquiredInProcessCost",
    )),
    Concept("income_tax", "Income tax expense", DURATION, (
        "IncomeTaxExpenseBenefit",
    )),
    Concept("impairment", "Impairment charges", DURATION, (
        "GoodwillAndIntangibleAssetImpairment",
        "GoodwillImpairmentLoss",
        "AssetImpairmentCharges",
        "ImpairmentOfLongLivedAssetsHeldForUse",
        "ImpairmentOfIntangibleAssetsExcludingGoodwill",
    ), pick="largest"),
    Concept("restructuring", "Restructuring charges", DURATION, (
        "RestructuringCharges",
        "RestructuringCostsAndAssetImpairmentCharges",
        "RestructuringSettlementAndImpairmentProvisions",
    )),
    Concept("diluted_shares", "Diluted shares (weighted avg.)", DURATION, (
        "WeightedAverageNumberOfDilutedSharesOutstanding",
        "WeightedAverageNumberOfSharesOutstandingBasic",
    ), unit="shares"),
    # ---- cash flow ----------------------------------------------------------
    Concept("operating_cash_flow", "Cash from operations", DURATION, (
        "NetCashProvidedByUsedInOperatingActivities",
        "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations",
    ), statement="CF"),
    Concept("capex", "Capital expenditures", DURATION, (
        "PaymentsToAcquirePropertyPlantAndEquipment",
        "PaymentsToAcquireProductiveAssets",
        "PaymentsToAcquireOtherPropertyPlantAndEquipment",
    ), pick="largest", statement="CF"),
    Concept("sbc", "Stock-based compensation", DURATION, (
        "ShareBasedCompensation",
        "AllocatedShareBasedCompensationExpense",
    ), statement="CF"),
    Concept("dividends", "Dividends paid", DURATION, (
        "PaymentsOfDividends",
        "PaymentsOfDividendsCommonStock",
    ), statement="CF"),
    Concept("buybacks", "Share repurchases", DURATION, (
        "PaymentsForRepurchaseOfCommonStock",
    ), statement="CF"),
)

CONCEPT_BY_KEY = {c.key: c for c in CONCEPTS}

# tag -> (concept key, priority); lower priority number wins
TAG_TO_CONCEPT: dict[str, tuple[str, int]] = {}
for _c in CONCEPTS:
    for _i, _t in enumerate(_c.tags):
        TAG_TO_CONCEPT.setdefault(_t, (_c.key, _i))

MAPPED_TAGS = frozenset(TAG_TO_CONCEPT)
TAG_UNIT = {tag: CONCEPT_BY_KEY[key].unit for tag, (key, _) in TAG_TO_CONCEPT.items()}
