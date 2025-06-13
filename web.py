import gradio as gr
from gradio_client import Client
from api import run_model


# Dummy classifier function

# Classify input
def classify_input(description, category, state):
    if description:
        return run_model(description, category, state)
    return "Please fill in all required fields."

# Flag and return updated flagged list
def flag_result(description, category, state, result, flagged_data):
    # Validate result is non-empty and matches inputs
    expected_result = run_model(description, category, state)
    if not result or result.strip() != expected_result.strip():
        return gr.update(), flagged_data, "⚠️ Please click 'Submit' before flagging."

    flagged_data.append([description, category, state, result])
    return flagged_data, flagged_data, ""  # Clear error


# Clear all flagged examples
def clear_flagged():
    return [], [], ""


with gr.Blocks() as demo:
    gr.Markdown("## Automatic identification of responsible authorities for citizen appeals")

    with gr.Row():
        description_input = gr.Textbox(label="Description", lines=2)
        category_input = gr.Dropdown(choices=["Verkehr", "Bildung", "Migration", "Umwelt", "Gesundheit", "Wirtschaft", "Digitalisierung", "Sicherheit"], label="Category")
        state_input = gr.Dropdown(choices=['Sachsen', 'Niedersachsen', 'Thüringen', 'Bayern',
       'Baden-Württemberg', 'Nordrhein-Westfalen', 'Sachsen-Anhalt',
       'Brandenburg', 'Hessen', 'Rheinland-Pfalz',
       'Mecklenburg-Vorpommern', 'Schleswig-Holstein', 'Berlin',
       'Hamburg', 'Saarland', 'Bremen'], label="State")

    result_output = gr.Textbox(label="Model Output", interactive=False)
    error_output = gr.Markdown("")  # <-- New error display

    with gr.Row():
        submit_button = gr.Button("Submit")
        flag_button = gr.Button("Flag this result")
        clear_button = gr.Button("Clear All Flagged Results")

    flagged_table = gr.Dataframe(
        headers=["Description", "Category", "State", "Result"],
        label="Flagged Examples",
        interactive=False
    )

    flagged_state = gr.State([])

    # On Submit
    submit_button.click(
        fn=classify_input,
        inputs=[description_input, category_input, state_input],
        outputs=result_output,
        api_name="classify_input"
    )

    # On Flag
    flag_button.click(
        fn=flag_result,
        inputs=[description_input, category_input, state_input, result_output, flagged_state],
        outputs=[flagged_table, flagged_state, error_output],
        api_name="flag_result"
    )

    # On Clear
    clear_button.click(
        fn=clear_flagged,
        inputs=[],
        outputs=[flagged_table, flagged_state, error_output],
        api_name="clear_flagged"
    )



if __name__ == "__main__":
    demo.queue(api_open=True).launch()
    client = Client(demo.local_url)
    result = client.predict(
          description="Hello!!",
          category="Verkehr",
          state="Sachsen",
          api_name="/classify_input"
    )
