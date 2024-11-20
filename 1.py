        import seaborn  # scope for faster 'import ultralytics'

        classes_to_log = [0, 56, 57, 58]
        filtered_matrix = self.matrix[np.ix_(classes_to_log, classes_to_log + [self.nc])]

        array = filtered_matrix / ((filtered_matrix.sum(0).reshape(1, -1) + 1e-9) if normalize else 1)  # normalize columns
        array[array < 0.005] = np.nan  # don't annotate (would appear as 0.00)

        fig, ax = plt.subplots(1, 1, figsize=(12, 9), tight_layout=True)
        class_names = ["person", "chair", "couch", "potted plant"]
        seaborn.set_theme(font_scale=1.0 if len(class_names) < 50 else 0.8)  # for label size
        ticklabels = class_names + ["background"]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # suppress empty matrix RuntimeWarning: All-NaN slice encountered
            seaborn.heatmap(
            array,
            ax=ax,
            annot=len(class_names) < 30,
            annot_kws={"size": 8},
            cmap="Blues",
            fmt=".2f" if normalize else ".0f",
            square=True,
            vmin=0.0,
            xticklabels=ticklabels,
            yticklabels=ticklabels,
            ).set_facecolor((1, 1, 1))
        title = "Confusion Matrix" + " Normalized" * normalize
        ax.set_xlabel("True")
        ax.set_ylabel("Predicted")
        ax.set_title(title)
        plot_fname = Path(save_dir) / f'{title.lower().replace(" ", "_")}.png'
        fig.savefig(plot_fname, dpi=250)
        plt.close(fig)
        if on_plot:
            on_plot(plot_fname)
