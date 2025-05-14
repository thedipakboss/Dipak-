from manim import *
import numpy as np

class FourierSeriesVideo(Scene):
    def construct(self):
        # Set video duration to approximately 8–9 minutes (520 seconds)
        self.total_duration = 520
        self.setup_scene()
        self.introduction()
        self.explain_fourier_series()
        self.show_epicycles()
        self.square_wave_approximation()
        self.triangle_wave_approximation()
        self.sawtooth_wave_approximation()
        self.gibbs_phenomenon()
        self.fourier_transform_teaser()
        self.real_life_applications()
        self.summary()
        self.conclusion()

    def setup_scene(self):
        # Set a dark background for better contrast
        self.camera.background_color = DARK_GRAY

    def introduction(self):
        # Detailed Introduction Slide with Quote and Signal Animation
        title = Text("Understanding the Fourier Series", font_size=48, color=WHITE)
        presenter = Text("Presented by Dipak Rajendra Patil", font_size=32, color=BLUE)
        club = Text("for VERTEX", font_size=28, color=YELLOW)

        title.to_edge(UP, buff=1)
        presenter.next_to(title, DOWN, buff=0.5)
        club.next_to(presenter, DOWN, buff=0.5)

        self.play(
            Write(title, run_time=2.5),
            Write(presenter, run_time=2),
            Write(club, run_time=2)
        )
        self.wait(2)

        # Fourier quote
        quote = Text(
            "\"Mathematics compares the most diverse phenomena\nand discovers their hidden analogies.\" — Joseph Fourier",
            font_size=24, color=WHITE
        ).next_to(club, DOWN, buff=0.5)
        self.play(Write(quote), run_time=3)
        self.wait(3)

        # Brief history and significance
        history = VGroup(
            Text("Introduced by Joseph Fourier in 1822", font_size=24, color=WHITE),
            Text("Breaks down periodic functions into sines and cosines", font_size=24, color=WHITE),
            Text("Foundation for signal processing and modern technology", font_size=24, color=WHITE)
        ).arrange(DOWN, buff=0.3).next_to(quote, DOWN, buff=0.5)
        self.play(Write(history), run_time=3)
        self.wait(2)

        # Show a periodic wave and real-world signal
        axes = Axes(
            x_range=[0, 2*PI, PI/2],
            y_range=[-1.5, 1.5, 0.5],
            axis_config={"color": GREY},
            x_length=6,
            y_length=2
        ).next_to(history, DOWN, buff=0.5)
        wave = axes.plot(lambda x: np.sin(x) + 0.5*np.sin(3*x), color=BLUE)
        wave_label = Text("A Periodic Function (e.g., Audio Signal)", font_size=20, color=BLUE).next_to(axes, UP)
        self.play(Create(axes), Create(wave), Write(wave_label), run_time=2.5)
        self.wait(3)

        self.play(
            FadeOut(title),
            FadeOut(presenter),
            FadeOut(club),
            FadeOut(quote),
            FadeOut(history),
            FadeOut(axes),
            FadeOut(wave),
            FadeOut(wave_label)
        )

    def explain_fourier_series(self):
        # Section Title
        title = Text("What is a Fourier Series?", font_size=36, color=WHITE).to_edge(UP)
        self.play(Write(title), run_time=1.5)

        # Explain the concept with text
        explanation = VGroup(
            Text("Approximates periodic functions using:", font_size=24, color=WHITE),
            Text("f(x) = a₀/2 + Σ [aₙ cos(nωx) + bₙ sin(nωx)]", font_size=24, color=YELLOW),
            Text("aₙ, bₙ: Fourier coefficients, ω: fundamental frequency", font_size=24, color=WHITE)
        ).arrange(DOWN, buff=0.3).next_to(title, DOWN, buff=0.5)
        self.play(Write(explanation), run_time=3.5)
        self.wait(3)

        # Show a simple sine wave
        axes = Axes(
            x_range=[-PI, PI, PI/2],
            y_range=[-1.5, 1.5, 0.5],
            axis_config={"color": GREY},
            x_length=8,
            y_length=4
        ).next_to(explanation, DOWN, buff=0.5)
        sine_graph = axes.plot(lambda x: np.sin(x), color=BLUE)
        sine_label = Text("sin(x): A single Fourier term", font_size=20, color=BLUE).next_to(axes, UP)

        self.play(Create(axes), Create(sine_graph), Write(sine_label), run_time=2.5)
        self.wait(3)

        # Transition to a complex wave
        complex_graph = axes.plot(lambda x: np.sin(x) + 0.5 * np.sin(3*x), color=YELLOW)
        complex_label = Text("sin(x) + 0.5*sin(3x)", font_size=20, color=YELLOW).next_to(sine_label, DOWN)
        self.play(
            Transform(sine_graph, complex_graph),
            Transform(sine_label, complex_label),
            run_time=2.5
        )
        self.wait(3)

        self.play(
            FadeOut(axes),
            FadeOut(sine_graph),
            FadeOut(sine_label),
            FadeOut(explanation),
            FadeOut(title)
        )

    def show_epicycles(self):
        # Section Title
        title = Text("Fourier Series as Epicycles", font_size=36, color=WHITE).to_edge(UP)
        self.play(Write(title), run_time=1.5)

        # Explanation
        explanation = Text(
            "Each Fourier term is a rotating vector (epicycle)\nRadius = amplitude, Frequency = nω",
            font_size=24, color=WHITE
        ).next_to(title, DOWN, buff=0.5)
        self.play(Write(explanation), run_time=2.5)
        self.wait(2)

        # First epicycle animation (5 circles)
        num_circles = 5
        center = np.array([0, 0, 0])
        base_radius = 1.5
        circles = VGroup()
        lines = VGroup()
        labels = VGroup()

        current_point = center
        for i in range(num_circles):
            r = base_radius / (i + 1)
            circle_center = current_point + np.array([r, 0, 0])
            circle = Circle(radius=r, color=BLUE, stroke_opacity=0.5).move_to(circle_center)
            line = Line(current_point, circle_center, color=WHITE, stroke_width=3)
            label = Text(f"r={r:.2f}, f={i+1}", font_size=16, color=YELLOW).next_to(circle, UP, buff=0.1)
            circles.add(circle)
            lines.add(line)
            labels.add(label)
            current_point = circle_center

        all_items = VGroup(circles, lines, labels).move_to(DOWN * 1)
        self.play(*[Create(c) for c in circles], *[Create(l) for l in lines], *[Write(l) for l in labels], run_time=3.5)

        initial_point = lines[-1].get_end()
        path = VMobject(color=ORANGE, stroke_width=4)
        path.set_points_as_corners([initial_point, initial_point])

        time_tracker = ValueTracker(0)
        def update_circles_and_lines(mob):
            time = time_tracker.get_value()
            new_points = [center]
            for i in range(num_circles):
                r = base_radius / (i + 1)
                angle = (i + 1) * time
                new_center = new_points[-1] + r * np.array([np.cos(angle), np.sin(angle), 0])
                circles[i].move_to(new_center)
                lines[i].put_start_and_end_on(new_points[-1], new_center)
                labels[i].next_to(circles[i], UP, buff=0.1)
                new_points.append(new_center)
            if len(new_points) > 1:
                path.add_line_to(new_points[-1])

        all_items.add_updater(update_circles_and_lines)
        self.add(all_items, path)
        self.play(time_tracker.animate.set_value(2 * PI), run_time=6, rate_func=linear)
        self.wait(3)

        # Clean up first animation
        all_items.remove_updater(update_circles_and_lines)
        self.play(FadeOut(circles), FadeOut(lines), FadeOut(labels), FadeOut(path))

        # Second epicycle animation (3 circles, different configuration)
        num_circles = 3
        circles = VGroup()
        lines = VGroup()
        labels = VGroup()
        current_point = center
        for i in range(num_circles):
            r = base_radius / (i + 1.5)  # Slightly different radius scaling
            circle_center = current_point + np.array([r, 0, 0])
            circle = Circle(radius=r, color=GREEN, stroke_opacity=0.5).move_to(circle_center)
            line = Line(current_point, circle_center, color=WHITE, stroke_width=3)
            label = Text(f"r={r:.2f}, f={i+1}", font_size=16, color=YELLOW).next_to(circle, UP, buff=0.1)
            circles.add(circle)
            lines.add(line)
            labels.add(label)
            current_point = circle_center

        all_items = VGroup(circles, lines, labels).move_to(DOWN * 1)
        self.play(*[Create(c) for c in circles], *[Create(l) for l in lines], *[Write(l) for l in labels], run_time=3.5)

        initial_point = lines[-1].get_end()
        path = VMobject(color=RED, stroke_width=4)
        path.set_points_as_corners([initial_point, initial_point])

        time_tracker = ValueTracker(0)
        all_items.add_updater(update_circles_and_lines)
        self.add(all_items, path)
        self.play(time_tracker.animate.set_value(2 * PI), run_time=5, rate_func=linear)
        self.wait(3)

        # Clean up
        all_items.remove_updater(update_circles_and_lines)
        self.play(
            FadeOut(circles),
            FadeOut(lines),
            FadeOut(labels),
            FadeOut(path),
            FadeOut(explanation),
            FadeOut(title)
        )

    def square_wave_approximation(self):
        # Section Title
        title = Text("Square Wave Approximation", font_size=36, color=WHITE).to_edge(UP)
        self.play(Write(title), run_time=1.5)

        # Explanation
        explanation = Text(
            "Approximates a square wave using odd sine terms",
            font_size=24, color=WHITE
        ).next_to(title, DOWN, buff=0.5)
        self.play(Write(explanation), run_time=2.5)

        # Equation
        equation = Text(
            "f(x) = (4/π) Σ (sin((2k-1)x) / (2k-1)), k=1 to ∞",
            font_size=24, color=YELLOW
        ).next_to(explanation, DOWN, buff=0.5)
        self.play(Write(equation), run_time=2.5)

        # Axes setup
        axes = Axes(
            x_range=[-PI, PI, PI/2],
            y_range=[-1.5, 1.5, 0.5],
            axis_config={"color": GREY},
            x_length=8,
            y_length=4
        ).next_to(explanation, DOWN, buff=0.5)
        self.play(Create(axes), run_time=1.5)

        # Square wave function
        def square_wave(x, n_terms):
            return sum((4 / (np.pi * (2*k - 1))) * np.sin((2*k - 1) * x) for k in range(1, n_terms + 1))

        # Animate increasing terms
        for n in [1, 3, 5, 11]:
            graph = axes.plot(lambda x: square_wave(x, n), color=YELLOW, use_smoothing=False)
            label = Text(f"n = {n} terms", font_size=20, color=YELLOW).next_to(axes, UP)
            self.play(Create(graph), Write(label), run_time=2)
            self.wait(3)
            self.play(FadeOut(graph), FadeOut(label))

        self.play(FadeOut(axes), FadeOut(equation), FadeOut(explanation), FadeOut(title))

    def triangle_wave_approximation(self):
        # Section Title
        title = Text("Triangle Wave Approximation", font_size=36, color=WHITE).to_edge(UP)
        self.play(Write(title), run_time=1.5)

        # Explanation
        explanation = Text(
            "Approximates a triangle wave using odd cosine terms",
            font_size=24, color=WHITE
        ).next_to(title, DOWN, buff=0.5)
        self.play(Write(explanation), run_time=2.5)

        # Equation
        equation = Text(
            "f(x) = (8/π²) Σ ((-1)^((k-1)/2) / k²) cos(kx), k odd",
            font_size=24, color=TEAL
        ).next_to(explanation, DOWN, buff=0.5)
        self.play(Write(equation), run_time=2.5)

        # Axes setup
        axes = Axes(
            x_range=[-PI, PI, PI/2],
            y_range=[-1.5, 1.5, 0.5],
            axis_config={"color": GREY},
            x_length=8,
            y_length=4
        ).next_to(explanation, DOWN, buff=0.5)
        self.play(Create(axes), run_time=1.5)

        # Triangle wave function
        def triangle_wave(x, n_terms):
            return sum(
                (8 * (-1)**((k-1)//2) / (np.pi**2 * k**2)) * np.cos(k * x)
                for k in range(1, n_terms + 1, 2)
            )

        # Animate increasing terms
        for n in [1, 3, 5, 11]:
            graph = axes.plot(lambda x: triangle_wave(x, n), color=TEAL, use_smoothing=False)
            label = Text(f"n = {n} terms", font_size=20, color=TEAL).next_to(axes, UP)
            self.play(Create(graph), Write(label), run_time=2)
            self.wait(3)
            self.play(FadeOut(graph), FadeOut(label))

        self.play(FadeOut(axes), FadeOut(equation), FadeOut(explanation), FadeOut(title))

    def sawtooth_wave_approximation(self):
        # Section Title
        title = Text("Sawtooth Wave Approximation", font_size=36, color=WHITE).to_edge(UP)
        self.play(Write(title), run_time=1.5)

        # Explanation
        explanation = Text(
            "Approximates a sawtooth wave using sine terms",
            font_size=24, color=WHITE
        ).next_to(title, DOWN, buff=0.5)
        self.play(Write(explanation), run_time=2.5)

        # Equation
        equation = Text(
            "f(x) = (2/π) Σ ((-1)^k sin(kx) / k), k=1 to ∞",
            font_size=24, color=PURPLE
        ).next_to(explanation, DOWN, buff=0.5)
        self.play(Write(equation), run_time=2.5)

        # Axes setup
        axes = Axes(
            x_range=[-PI, PI, PI/2],
            y_range=[-1.5, 1.5, 0.5],
            axis_config={"color": GREY},
            x_length=8,
            y_length=4
        ).next_to(explanation, DOWN, buff=0.5)
        self.play(Create(axes), run_time=1.5)

        # Sawtooth wave function
        def sawtooth_wave(x, n_terms):
            return sum((2 * (-1)**k / (np.pi * k)) * np.sin(k * x) for k in range(1, n_terms + 1))

        # Animate increasing terms
        for n in [1, 3, 5, 11]:
            graph = axes.plot(lambda x: sawtooth_wave(x, n), color=PURPLE, use_smoothing=False)
            label = Text(f"n = {n} terms", font_size=20, color=PURPLE).next_to(axes, UP)
            self.play(Create(graph), Write(label), run_time=2)
            self.wait(3)
            self.play(FadeOut(graph), FadeOut(label))

        self.play(FadeOut(axes), FadeOut(equation), FadeOut(explanation), FadeOut(title))

    def gibbs_phenomenon(self):
        # Section Title
        title = Text("Gibbs Phenomenon", font_size=36, color=WHITE).to_edge(UP)
        self.play(Write(title), run_time=1.5)

        # Explanation
        explanation = Text(
            "Overshooting occurs at discontinuities\nin Fourier approximations (e.g., square wave)",
            font_size=24, color=WHITE
        ).next_to(title, DOWN, buff=0.5)
        self.play(Write(explanation), run_time=2.5)

        # Axes setup
        axes = Axes(
            x_range=[-PI, PI, PI/2],
            y_range=[-1.5, 1.5, 0.5],
            axis_config={"color": GREY},
            x_length=8,
            y_length=4
        ).next_to(explanation, DOWN, buff=0.5)
        self.play(Create(axes), run_time=1.5)

        # Square wave with high terms to show Gibbs
        def square_wave(x, n_terms):
            return sum((4 / (np.pi * (2*k - 1))) * np.sin((2*k - 1) * x) for k in range(1, n_terms + 1))

        graph = axes.plot(lambda x: square_wave(x, 50), color=RED, use_smoothing=False)
        label = Text("n = 50 terms (overshooting at edges)", font_size=20, color=RED).next_to(axes, UP)
        self.play(Create(graph), Write(label), run_time=2.5)
        self.wait(3)

        self.play(FadeOut(axes), FadeOut(graph), FadeOut(label), FadeOut(explanation), FadeOut(title))

    def fourier_transform_teaser(self):
        # Section Title
        title = Text("Fourier Series to Fourier Transform", font_size=36, color=WHITE).to_edge(UP)
        self.play(Write(title), run_time=1.5)

        # Explanation
        explanation = Text(
            "Fourier Series for periodic functions\nFourier Transform for non-periodic signals",
            font_size=24, color=WHITE
        ).next_to(title, DOWN, buff=0.5)
        self.play(Write(explanation), run_time=2.5)

        # Simple frequency spectrum animation
        axes = Axes(
            x_range=[0, 5, 1],
            y_range=[0, 1, 0.25],
            axis_config={"color": GREY},
            x_length=6,
            y_length=3
        ).next_to(explanation, DOWN, buff=0.5)
        bars = VGroup(*[
            Rectangle(height=0.8/(i+1), width=0.4, fill_color=YELLOW, fill_opacity=0.8)
            .move_to(axes.c2p(i+1, 0.4/(i+1)))
            for i in range(4)
        ])
        self.play(Create(axes), Create(bars), run_time=2.5)
        self.wait(3)

        self.play(FadeOut(axes), FadeOut(bars), FadeOut(explanation), FadeOut(title))

    def real_life_applications(self):
        # Section Title
        title = Text("Applications of Fourier Series", font_size=36, color=WHITE).to_edge(UP)
        self.play(Write(title), run_time=1.5)

        # List of applications with images
        apps = [
            (
                "Audio Processing",
                "Decomposes audio into frequencies for noise filtering and MP3 compression.",
                "audio_processing.png"
            ),
            (
                "Image Compression",
                "JPEG uses Discrete Cosine Transform (DCT), a Fourier-based method, to reduce file sizes.",
                "image_compression.png"
            ),
            (
                "Medical Imaging",
                "MRI scans reconstruct images from frequency data using Fourier Series.",
                "medical_imaging.png"
            ),
            (
                "Telecommunications",
                "Modulates signals into frequency components for wireless transmission.",
                "telecommunications.png"
            ),
            (
                "Vibration Analysis",
                "Analyzes vibrations in frequency domain to detect faults in machines.",
                "vibration_analysis.png"
            ),
        ]

        for app_name, desc, img_file in apps:
            # Image instead of placeholder
            image = ImageMobject(img_file).scale_to_fit_height(2).to_edge(LEFT, buff=1)
            app_label = Text(app_name, font_size=28, color=YELLOW).next_to(image, UP)
            app_desc = Text(desc, font_size=20, color=WHITE).next_to(image, RIGHT, buff=0.5).set_width(5)

            self.play(FadeIn(image), Write(app_label), Write(app_desc), run_time=2.5)
            self.wait(4)
            self.play(FadeOut(image), FadeOut(app_label), FadeOut(app_desc))

        self.play(FadeOut(title))

    def summary(self):
        # Summary Slide
        title = Text("Key Takeaways", font_size=36, color=WHITE).to_edge(UP)
        points = VGroup(
            Text("• Fourier Series decomposes periodic functions", font_size=24, color=WHITE),
            Text("• Uses sines and cosines with varying frequencies", font_size=24, color=WHITE),
            Text("• Visualized as epicycles or wave approximations", font_size=24, color=WHITE),
            Text("• Explains phenomena like Gibbs overshooting", font_size=24, color=WHITE),
            Text("• Powers audio, imaging, and more", font_size=24, color=WHITE)
        ).arrange(DOWN, buff=0.3, aligned_edge=LEFT).next_to(title, DOWN, buff=0.5)

        self.play(Write(title), Write(points), run_time=3.5)
        self.wait(4)
        self.play(FadeOut(title), FadeOut(points))

    def conclusion(self):
        # Thank You Slide with Call-to-Action, properly aligned
        thank_you = Text("Thank You!", font_size=48, color=WHITE).set_color_by_gradient(YELLOW, ORANGE)
        final_note = Text("Explore the math that shapes our world!", font_size=28, color=WHITE)
        cta = Text("Learn more: Study signal processing or Fourier analysis!", font_size=26, color=BLUE)

        # Center text elements vertically
        text_group = VGroup(thank_you, final_note, cta).arrange(DOWN, buff=0.5).move_to(UP * 1)

        # Rotating circles below text
        circle1 = Circle(radius=1, color=BLUE).move_to(DOWN * 3 + LEFT * 2)
        circle2 = Circle(radius=0.5, color=GREEN).move_to(DOWN * 3 + RIGHT * 2)
        dot1 = Dot(color=YELLOW).move_to(circle1.point_from_proportion(0))
        dot2 = Dot(color=RED).move_to(circle2.point_from_proportion(0))

        self.play(
            FadeIn(text_group, run_time=2),
            Create(circle1, run_time=4),
            Create(circle2, run_time=4),
            Create(dot1, run_time=4),
            Create(dot2, run_time=4)
        )
        self.play(
            Rotate(dot1, angle=2*PI, about_point=circle1.get_center(), run_time=5),
            Rotate(dot2, angle=-2*PI, about_point=circle2.get_center(), run_time=5)
        )
        self.wait(3)
        self.play(
            FadeOut(text_group),
            FadeOut(circle1),
            FadeOut(circle2),
            FadeOut(dot1),
            FadeOut(dot2)
        )