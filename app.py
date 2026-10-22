import os
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(
    page_title="ReEngage AI",
    page_icon="🛍️",
    layout="wide"
)

st.title("🛍️ ReEngage AI")
st.caption("Turn customer data into personalised retention journeys.")

REQUIRED_COLUMNS = [
    "customer_id",
    "customer_name",
    "last_purchase_days",
    "total_orders",
    "total_spend",
    "last_product",
    "email_consent",
    "previous_response"
]

SEGMENT_COLORS = {
    "Inactive": "#E05252",
    "At risk": "#D9A21B",
    "Loyal": "#299764",
    "New customer": "#55A9E8",
}

# Each segment has a suggested strategy and matching campaign objective.
SEGMENT_GUIDANCE = {
    "New customer": {
        "reason": "The customer has made one recorded order.",
        "strategy": "Encourage a second purchase",
        "objective": "Encourage a second purchase"
    },
    "Loyal": {
        "reason": "The customer has made at least two orders and purchased within the last 30 days.",
        "strategy": "Reward customer loyalty",
        "objective": "Reward customer loyalty"
    },
    "At risk": {
        "reason": "The last purchase was 31–60 days ago.",
        "strategy": "Recommend relevant products",
        "objective": "Recommend relevant products"
    },
    "Inactive": {
        "reason": "The last purchase was more than 60 days ago.",
        "strategy": "Win-back the inactive customer",
        "objective": "Reactivate an inactive customer"
    }
}

OBJECTIVES = [
    "Encourage a second purchase",
    "Reward customer loyalty",
    "Recommend relevant products",
    "Reactivate an inactive customer"
]


def load_sample_data():
    path = Path(__file__).parent / "customers.csv"
    return pd.read_csv(path)


def assign_segment(customer):
    days = int(customer["last_purchase_days"])
    orders = int(customer["total_orders"])

    # Recency takes priority: a long-inactive customer should not be labelled new
    # just because they have only one recorded order.
    if days > 60:
        return "Inactive"
    elif days > 30:
        return "At risk"
    elif orders == 1:
        return "New customer"
    else:
        # Any customer with 2+ orders who purchased within 30 days is loyal.
        return "Loyal"


def get_api_key():
    key = os.getenv("OPENAI_API_KEY", "")

    if not key:
        try:
            key = st.secrets.get("OPENAI_API_KEY", "")
        except Exception:
            key = ""

    return key


def generate_message(customer, objective):
    prompt = f"""
You are a helpful retail customer retention assistant.

Write one concise marketing message using only the customer information below.

Customer first name: {customer['customer_name']}
Customer segment: {customer['segment']}
Days since last purchase: {customer['last_purchase_days']}
Number of previous orders: {customer['total_orders']}
Previous product: {customer['last_product']}
Previous response: {customer['previous_response']}
Campaign objective: {objective}

Rules:
- Match the wording to the campaign objective.
- For "Reward customer loyalty", thank the customer
  for being a repeat customer and include a gentle call
  to action to explore the product range again.
- Do not invent purchase history or personal preferences.
- Do not claim that a discount or reward exists unless provided.
- Do not include sensitive personal information.
- Do not promise results.
- Keep the message under 90 words.
- Do not include an email address or pretend the message was sent.
- Return the message only.
"""

    api_key = get_api_key()

    if api_key:
        try:
            from openai import OpenAI

            client = OpenAI(api_key=api_key)
            response = client.responses.create(
                model=os.getenv("OPENAI_MODEL", "gpt-5.5"),
                input=prompt
            )
            return response.output_text.strip(), "AI-generated"
        except Exception:
            st.warning(
                "AI generation failed. A template message will be used instead. "
                "Check your API key and model settings."
            )

    name = customer["customer_name"]
    product = customer["last_product"]

    if objective == "Recommend relevant products":
        message = (
            f"Hi {name}! We thought you might like to explore more products "
            f"from our {product} range. Take a look at our collection "
            "whenever you're ready. Thanks for shopping with us!"
        )
    elif objective == "Encourage a second purchase":
        message = (
            f"Hi {name}! Thanks for shopping with us before. We'd love to "
            f"welcome you back to explore our {product} collection. "
            "Visit us again whenever you're ready!"
        )
    elif objective == "Reward customer loyalty":
        message = (
            f"Hi {name}! Thank you for choosing us for your recent purchases. "
            "We appreciate your loyalty! As a returning customer, we'd love "
            f"for you to explore our {product} collection again and discover "
            "something you may enjoy next."
        )
    elif objective == "Reactivate an inactive customer":
        message = (
            f"Hi {name}! It's been a while since your last visit, and we'd "
            f"love to welcome you back. Feel free to explore our {product} "
            "collection whenever you're ready."
        )
    else:
        message = (
            f"Hi {name}! Thanks for being part of our customer community. "
            f"If you're interested, you can explore more from our {product} "
            "collection. We hope to see you again soon!"
        )

    return message, "Template fallback"


# Initialise application data.
if "outcomes" not in st.session_state:
    st.session_state.outcomes = {}

if "messages" not in st.session_state:
    st.session_state.messages = {}

if "message_versions" not in st.session_state:
    st.session_state.message_versions = {}

# Session-level suppression list for customers who opt out during this demo.
if "opted_out_customers" not in st.session_state:
    st.session_state.opted_out_customers = set()

with st.sidebar:
    st.header("Customer data")
    uploaded_file = st.file_uploader(
        "Upload a customer CSV",
        type=["csv"]
    )
    st.caption(
        "Use fictional or appropriately authorised data. "
        "Marketing eligibility is checked before engagement."
    )

try:
    if uploaded_file is not None:
        raw_df = pd.read_csv(uploaded_file)
    else:
        raw_df = load_sample_data()

    missing = [column for column in REQUIRED_COLUMNS if column not in raw_df.columns]
    if missing:
        st.error("Missing required columns: " + ", ".join(missing))
        st.stop()

    df = raw_df[REQUIRED_COLUMNS].copy()
    numeric_columns = ["last_purchase_days", "total_orders", "total_spend"]

    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    if df[numeric_columns].isna().any().any():
        st.error("Some numeric fields are missing or invalid.")
        st.stop()

    if (
        (df["last_purchase_days"] < 0).any()
        or (df["total_orders"] < 1).any()
        or (df["total_spend"] < 0).any()
    ):
        st.error("Check days, order counts and spending values.")
        st.stop()

    df["email_consent"] = (
        df["email_consent"].astype(str).str.strip().str.lower()
        .isin(["true", "1", "yes"])
    )

    if df["customer_id"].duplicated().any():
        st.error("Customer IDs must be unique.")
        st.stop()

    if df["customer_name"].isna().any():
        st.error("Customer names cannot be empty.")
        st.stop()

    df["segment"] = df.apply(assign_segment, axis=1)

except Exception:
    st.error(
        "Could not load the data. Check that customers.csv exists "
        "and follows the required format."
    )
    st.stop()


# Main navigation.
overview_tab, customers_tab, journey_tab, results_tab = st.tabs(
    ["📊 Overview", "👥 Customers", "💬 Retention Journey", "📈 Campaign Results"]
)

with overview_tab:
    st.subheader("Customer intelligence")

    total_customers = len(df)
    at_risk_count = int(df["segment"].isin(["At risk", "Inactive"]).sum())
    eligible_count = int(df["email_consent"].sum())

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total customers", total_customers)
    col2.metric("At-risk / inactive", at_risk_count)
    col3.metric("Marketing eligible", eligible_count)
    col4.metric("Total customer spend", f"${df['total_spend'].sum():,.2f}")

    left, right = st.columns(2)

    with left:
        segment_counts = (
            df["segment"].value_counts().rename_axis("Segment")
            .reset_index(name="Customers")
        )
        fig = px.bar(
            segment_counts,
            x="Segment",
            y="Customers",
            color="Segment",
            color_discrete_map=SEGMENT_COLORS,
            category_orders={"Segment": list(SEGMENT_COLORS.keys())},
            title="Customer Segments"
        )
        fig.update_layout(showlegend=False, xaxis_title="Segment", yaxis_title="Customers")
        st.plotly_chart(fig, use_container_width=True)

    with right:
        fig = px.scatter(
            df,
            x="last_purchase_days",
            y="total_spend",
            color="segment",
            color_discrete_map=SEGMENT_COLORS,
            category_orders={"segment": list(SEGMENT_COLORS.keys())},
            hover_data=["customer_id", "customer_name"],
            title="Purchase Recency vs Total Spend",
            labels={
                "last_purchase_days": "Days Since Last Purchase",
                "total_spend": "Total Spend ($)",
                "segment": "Segment"
            }
        )
        st.plotly_chart(fig, use_container_width=True)

    st.info(
        "Segments use transparent rules, not a trained churn prediction model. "
        "Spending is historical, not predicted."
    )


with customers_tab:
    st.subheader("Explore customer records")

    selected_segment = st.selectbox(
        "Filter by segment",
        ["All"] + sorted(df["segment"].unique().tolist())
    )
    filtered_df = df.copy()

    if selected_segment != "All":
        filtered_df = filtered_df[filtered_df["segment"] == selected_segment]

    st.dataframe(filtered_df, use_container_width=True, hide_index=True)
    st.download_button(
        "Download customer segment report",
        data=filtered_df.to_csv(index=False).encode("utf-8"),
        file_name="customer_segments.csv",
        mime="text/csv"
    )


with journey_tab:
    st.subheader("Personalised Retention Journey")

    customer_ids = df["customer_id"].tolist()
    chosen_id = st.selectbox(
        "Select a demo customer",
        customer_ids,
        format_func=lambda customer_id: (
            customer_id + " — " + str(df.loc[df["customer_id"] == customer_id, "customer_name"].iloc[0])
        )
    )
    customer = df.loc[df["customer_id"] == chosen_id].iloc[0]
    segment = customer["segment"]
    guidance = SEGMENT_GUIDANCE[segment]
    segment_color = SEGMENT_COLORS[segment]

    st.markdown(
        f"""
        <style>
        .journey-card {{
            border: 1px solid rgba(128,128,128,0.35);
            border-radius: 14px;
            padding: 16px 18px;
            background: rgba(128,128,128,0.07);
            min-height: 94px;
        }}
        .journey-label {{ font-size: 0.88rem; opacity: 0.82; margin-bottom: 7px; }}
        .journey-value {{ font-size: 1.25rem; font-weight: 700; }}
        .decision-card {{
            border: 1px solid rgba(128,128,128,0.35);
            border-radius: 14px;
            padding: 17px 20px;
            background: rgba(128,128,128,0.06);
            margin-top: 8px;
            margin-bottom: 12px;
        }}
        .decision-heading {{ font-size: 1.05rem; font-weight: 700; margin-bottom: 6px; }}
        .decision-copy {{ font-size: 1.05rem; line-height: 1.5; }}
        </style>
        <div style="display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px;">
          <div class="journey-card">
            <div class="journey-label">Customer Segment</div>
            <div class="journey-value" style="color:{segment_color};">{segment}</div>
          </div>
          <div class="journey-card">
            <div class="journey-label">Days Since Purchase</div>
            <div class="journey-value">{int(customer['last_purchase_days'])}</div>
          </div>
          <div class="journey-card">
            <div class="journey-label">Previous Orders</div>
            <div class="journey-value">{int(customer['total_orders'])}</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("### Decision Explanation")
    st.markdown(
        f"""
        <div class="decision-card">
          <div class="decision-heading">Why Selected</div>
          <div class="decision-copy">{guidance['reason']}</div>
        </div>
        <div class="decision-card">
          <div class="decision-heading">Suggested Strategy</div>
          <div class="decision-copy">{guidance['strategy']}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    opted_out_in_session = chosen_id in st.session_state.opted_out_customers
    if not customer["email_consent"] or opted_out_in_session:
        if opted_out_in_session:
            st.warning(
                "This customer opted out during this demo session. "
                "Marketing is now suppressed for this customer, including message generation."
            )
        else:
            st.warning(
                "Marketing is suppressed because this customer has not consented. "
                "Message generation and journey execution are disabled for this customer."
            )
    else:
        default_objective = guidance["objective"]
        objective = st.selectbox(
            "Campaign Objective (auto-selected from the suggested strategy)",
            OBJECTIVES,
            index=OBJECTIVES.index(default_objective),
            key=f"objective_{chosen_id}"
        )
        saved_message = st.session_state.messages.get(chosen_id)
        generate_label = (
            "Regenerate Personalised Message"
            if saved_message else "Generate Personalised Message"
        )

        if st.button(generate_label, key=f"generate_{chosen_id}"):
            message, source = generate_message(customer, objective)
            st.session_state.messages[chosen_id] = {
                "text": message,
                "source": source,
                "objective": objective
            }
            st.session_state.message_versions[chosen_id] = (
                st.session_state.message_versions.get(chosen_id, 0) + 1
            )
            st.rerun()

        saved_message = st.session_state.messages.get(chosen_id)
        if saved_message:
            st.markdown("### Message Preview")
            st.caption("Generation method: " + saved_message["source"])
            message_version = st.session_state.message_versions.get(chosen_id, 1)
            edited_message = st.text_area(
                "Edit before simulated delivery",
                value=saved_message["text"],
                key=f"message_{chosen_id}_{message_version}",
                height=140
            )

            st.markdown("### Simulate Customer Outcome")
            st.caption("These buttons record demo outcomes only. No message is actually sent.")
            col1, col2, col3, col4 = st.columns(4)

            if col1.button("Clicked", key="click_" + chosen_id):
                st.session_state.outcomes[chosen_id] = {"status": "Clicked", "message": edited_message}
                st.success("Click recorded as a simulation.")
            if col2.button("Purchased", key="buy_" + chosen_id):
                st.session_state.outcomes[chosen_id] = {"status": "Purchased", "message": edited_message}
                st.success("Purchase recorded as a simulation.")
            if col3.button("No Response", key="none_" + chosen_id):
                st.session_state.outcomes[chosen_id] = {"status": "No response", "message": edited_message}
                st.info("No response recorded as a simulation.")
            if col4.button("Opted Out", key="out_" + chosen_id):
                st.session_state.outcomes[chosen_id] = {"status": "Opted out", "message": edited_message}
                st.session_state.opted_out_customers.add(chosen_id)
                st.success(
                    "Opt-out recorded. This customer is suppressed from further "
                    "message generation for the rest of this demo session."
                )
                st.rerun()


with results_tab:
    st.subheader("Campaign Results")
    st.caption("All outcomes and rates shown here are simulated for the prototype demo.")

    # Ignore old Exit Journey records from earlier versions of the prototype.
    # Exit Journey is no longer an available simulated outcome.
    outcomes = {
        customer_id: outcome
        for customer_id, outcome in st.session_state.outcomes.items()
        if outcome.get("status") not in ["Exited journey", "Exited"]
    }
    st.session_state.outcomes = outcomes

    outcome_order = ["Clicked", "Purchased", "No response", "Opted out"]
    outcome_counts = {status: 0 for status in outcome_order}
    for outcome in outcomes.values():
        status = outcome.get("status")
        if status in outcome_counts:
            outcome_counts[status] += 1

    total_recorded = sum(outcome_counts.values())
    clicked = outcome_counts["Clicked"]
    purchases = outcome_counts["Purchased"]
    no_response = outcome_counts["No response"]
    opted_out = outcome_counts["Opted out"]
    responses = clicked + purchases + opted_out

    # Shared styling for metric cards, matching the rounded cards in Retention Journey.
    st.markdown(
        """
        <style>
        .results-metric-card {
            border: 1px solid rgba(128,128,128,0.35);
            border-radius: 14px;
            padding: 16px 15px;
            background: rgba(128,128,128,0.07);
            min-height: 105px;
            box-sizing: border-box;
        }
        .results-metric-label {
            font-size: 0.88rem;
            opacity: 0.82;
            margin-bottom: 9px;
            line-height: 1.35;
        }
        .results-metric-value {
            font-size: 1.65rem;
            font-weight: 750;
            line-height: 1.2;
        }
        .results-rate-card {
            border: 1px solid rgba(128,128,128,0.35);
            border-radius: 14px;
            padding: 18px 20px;
            background: rgba(128,128,128,0.07);
            min-height: 120px;
            box-sizing: border-box;
        }
        .results-rate-label {
            font-size: 0.95rem;
            opacity: 0.85;
            margin-bottom: 10px;
        }
        .results-rate-value {
            font-size: 1.9rem;
            font-weight: 750;
            line-height: 1.2;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    metric_cards = [
        ("Recorded Outcomes", total_recorded, "#64748B"),
        ("Clicks", clicked, "#55A9E8"),
        ("Purchases", purchases, "#299764"),
        ("No Responses", no_response, "#9873D4"),
        ("Opt-Outs", opted_out, "#D9A21B"),
    ]
    metric_html = '<div style="display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:12px;">'
    for label, value, accent in metric_cards:
        metric_html += (
            '<div class="results-metric-card">'
            f'<div class="results-metric-label">{label}</div>'
            f'<div class="results-metric-value" style="color:{accent};">{value}</div>'
            '</div>'
        )
    metric_html += '</div>'
    st.markdown(metric_html, unsafe_allow_html=True)

    purchase_rate = f"{purchases / total_recorded * 100:.1f}%" if total_recorded else "—"
    response_rate = f"{responses / total_recorded * 100:.1f}%" if total_recorded else "—"
    rate_cards = [
        ("Simulated Purchase Conversion Rate", purchase_rate, "#299764"),
        ("Simulated Response Rate", response_rate, "#55A9E8"),
    ]
    rate_html = '<div style="display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:12px;margin-top:12px;">'
    for label, value, accent in rate_cards:
        rate_html += (
            '<div class="results-rate-card">'
            f'<div class="results-rate-label">{label}</div>'
            f'<div class="results-rate-value" style="color:{accent};">{value}</div>'
            '</div>'
        )
    rate_html += '</div>'
    st.markdown(rate_html, unsafe_allow_html=True)
    st.caption(
        "Purchase conversion rate = simulated purchases ÷ all recorded outcomes. "
        "Response rate = simulated clicks, purchases, or opt-outs ÷ all recorded outcomes. "
        "These are demo calculations, not real campaign performance."
    )

    results_df = pd.DataFrame({
        "Outcome": outcome_order,
        "Customers": [outcome_counts[status] for status in outcome_order]
    })
    outcome_fig = px.bar(
        results_df,
        x="Outcome",
        y="Customers",
        color="Outcome",
        color_discrete_map={
            "Clicked": "#55A9E8",
            "Purchased": "#299764",
            "No response": "#9873D4",
            "Opted out": "#D9A21B",
        },
        title="Simulated Customer Outcomes",
        category_orders={"Outcome": outcome_order}
    )
    outcome_fig.update_layout(
        showlegend=False,
        xaxis_title="Outcome",
        yaxis_title="Number of Customers",
        margin=dict(t=65, b=20, l=20, r=20),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(outcome_fig, use_container_width=True)

    if total_recorded > 0:
        st.markdown("### Recorded Outcome Details")
        details_df = pd.DataFrame([
            {"Customer ID": customer_id, "Outcome": outcome["status"]}
            for customer_id, outcome in outcomes.items()
            if outcome.get("status") in outcome_order
        ])
        st.dataframe(details_df, use_container_width=True, hide_index=True)
    else:
        st.info("Generate a message and record a simulated outcome in the Retention Journey tab to see demo results.")

    st.caption(
        "Demo outcomes are stored in session state and may reset when the app session restarts. "
        "No real purchases or campaign conversions are being measured."
    )
