import glob

import fortepyan as ff
import streamlit as st


def display_audio(title, midi_files: list[str], mp3_files: list[str]):
    st.title(title)

    cols = st.columns([2, 2, 2])
    fig_titles = ["### Original", "### Model", "### Model with EMA"]

    for i, col in enumerate(cols):
        with col:
            st.write(fig_titles[i])
            piece = ff.MidiFile(midi_files[i]).piece
            fig = ff.view.draw_pianoroll_with_velocities(piece)
            st.pyplot(fig)
            st.audio(mp3_files[i], format="audio/mp3")


def main():
    midi_files = glob.glob("tmp/midi/**")

    # get only filenames without extension, model type and dir
    filenames = [name[:-5].replace("-original", "").split("/")[-1] for name in midi_files if "original" in name]

    selected_filename = st.selectbox("Select piece to display", options=filenames)

    selected_midi = [
        f"tmp/midi/{selected_filename}-original.midi",
        f"tmp/midi/{selected_filename}-model.midi",
        f"tmp/midi/{selected_filename}-model-ema.midi",
    ]
    selected_mp3 = [
        f"tmp/mp3/{selected_filename}-original.mp3",
        f"tmp/mp3/{selected_filename}-model.mp3",
        f"tmp/mp3/{selected_filename}-model-ema.mp3",
    ]

    display_audio(
        title=selected_filename,
        midi_files=selected_midi,
        mp3_files=selected_mp3,
    )


if __name__ == "__main__":
    main()
