import streamlit as st
import subprocess
import os

def run_exercise_script(script_name):
    """Run the specified Python script and capture output."""
    try:
        result = subprocess.run(['python', script_name], capture_output=True, text=True)
        return result.stdout, result.stderr
    except Exception as e:
        return str(e), ""

def main():
    st.title("Muscle Mentor")

    # Select exercise
    exercise = st.selectbox(
        "Select Exercise",
        ["Bicep", "Tricep", "Shoulder", "Squad", "Push-Up"]  # Added Push-Up here
    )

    # Run selected exercise
    if st.button("Run Exercise"):
        if exercise == "Bicep":
            st.write("Running Bicep Curl Exercise...")
            script_path = os.path.join(os.path.dirname(__file__), 'Bicep.py')
            stdout, stderr = run_exercise_script(script_path)

        elif exercise == "Tricep":
            st.write("Running Tricep Exercise...")
            script_path = os.path.join(os.path.dirname(__file__), 'Tricep.py')
            stdout, stderr = run_exercise_script(script_path)

        elif exercise == "Shoulder":
            st.write("Running Shoulder Exercise...")
            script_path = os.path.join(os.path.dirname(__file__), 'Shoulder.py')
            stdout, stderr = run_exercise_script(script_path)

        elif exercise == "Squad":
            st.write("Running Squad Exercise...")
            script_path = os.path.join(os.path.dirname(__file__), 'Squad.py')
            stdout, stderr = run_exercise_script(script_path)

        elif exercise == "Push-Up":  # Added logic for Push-Up exercise
            st.write("Running Push-Up Exercise...")
            script_path = os.path.join(os.path.dirname(__file__), 'Pushup.py')  # Ensure the script name matches
            stdout, stderr = run_exercise_script(script_path)

            # Check if the script ended due to camera closure
            if "camera" in stderr.lower() or "camera" in stdout.lower():
                st.warning("The camera was closed or disconnected. Please reconnect the camera and try again.")

        else:
            st.write("Invalid exercise selected.")

        

if __name__ == "__main__":
    main()
