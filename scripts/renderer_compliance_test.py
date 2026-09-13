from pathlib import Path
import sys
import importlib
import numpy as np


# ============================================================================
# MICRONYX STEP 33B
# CANONICAL ACQUISITION MODEL COMPLIANCE TEST
#
# FINAL PS02 CANONICAL ACQUISITION MODEL
#
# Search:
#   1000 x 1000 pixels
#   5 pixels / physical unit
#
# Reference:
#   1000 x 1000 pixels
#   50 pixels / physical unit
#
# Therefore:
#   Reference sampling / Search sampling = 10x
#
# Search-equivalent template:
#   100 x 100 pixels
#
# Physical interpretation:
#
#   Search:
#       1000 / 5 = 200 x 200 physical units
#
#   Reference:
#       1000 / 50 = 20 x 20 physical units
#
#   Search-equivalent reference template:
#       20 physical units x 5 pixels/unit
#       = 100 x 100 pixels
#
# IMPORTANT:
#   The search and reference images intentionally have the same
#   pixel dimensions but different spatial sampling densities.
#
#   The reference is therefore 10x higher spatial resolution
#   while observing a smaller physical field of view.
#
# This test validates the canonical renderer and does not:
#   - generate new ground truth
#   - inject target fingerprints
#   - use an alternate renderer
#   - perform manual scene-specific selection
# ============================================================================


# ============================================================================
# EXPECTED CANONICAL MODEL
# ============================================================================

EXPECTED_SEARCH_SHAPE = (1000, 1000)

EXPECTED_REFERENCE_SHAPE = (1000, 1000)

EXPECTED_TEMPLATE_SHAPE = (100, 100)

EXPECTED_SEARCH_PPU = 5.0

EXPECTED_REFERENCE_PPU = 50.0

EXPECTED_SAMPLING_RATIO = 10.0


# ============================================================================
# TEST TARGET
# ============================================================================

TX = 75.25
TY = 113.75

SEED = 20260816

TOL = 1e-9


# ============================================================================
# CHECK HELPER
# ============================================================================

def check(
    condition,
    message,
    failures,
):
    """
    Record and print the result of an individual compliance check.
    """

    if condition:

        print(
            f"[PASS] {message}"
        )

    else:

        print(
            f"[FAIL] {message}"
        )

        failures.append(
            message
        )


# ============================================================================
# MAIN
# ============================================================================

def main():

    print()

    print(
        "=" * 76
    )

    print(
        "MICRONYX STEP 33B"
    )

    print(
        "CANONICAL ACQUISITION MODEL COMPLIANCE"
    )

    print(
        "=" * 76
    )

    print()


    # ========================================================================
    # IMPORT CANONICAL RENDERER
    # ========================================================================

    script_dir = Path(
        __file__
    ).resolve().parent

    if str(script_dir) not in sys.path:

        sys.path.insert(
            0,
            str(script_dir)
        )


    try:

        renderer = importlib.import_module(
            "canonical_renderer"
        )

    except Exception as exc:

        print(
            "[FAIL] Could not import canonical_renderer.py"
        )

        print(
            exc
        )

        return 1


    failures = []


    # ========================================================================
    # API CHECK
    # ========================================================================

    print(
        "API CHECK"
    )

    print(
        "-" * 76
    )


    required = [

        "periodic_scene",

        "quasiperiodic_scene",

        "continuous_scene",

        "render_search",

        "render_reference",

        "render_sensor",

        "create_ps02_template",

        "template_top_left",

        "generate_observation",

    ]


    for name in required:

        check(

            callable(
                getattr(
                    renderer,
                    name,
                    None,
                )
            ),

            f"{name}() exists",

            failures,

        )


    print()


    # ========================================================================
    # EXPLICIT ACQUISITION CONSTANTS
    # ========================================================================

    print(
        "ACQUISITION MODEL"
    )

    print(
        "-" * 76
    )


    search_ppu = getattr(

        renderer,

        "SEARCH_PIXELS_PER_UNIT",

        None,

    )


    reference_ppu = getattr(

        renderer,

        "REFERENCE_PIXELS_PER_UNIT",

        None,

    )


    # ------------------------------------------------------------------------
    # Search sampling constant
    # ------------------------------------------------------------------------

    check(

        search_ppu is not None,

        "SEARCH_PIXELS_PER_UNIT is explicitly defined",

        failures,

    )


    # ------------------------------------------------------------------------
    # Reference sampling constant
    # ------------------------------------------------------------------------

    check(

        reference_ppu is not None,

        "REFERENCE_PIXELS_PER_UNIT is explicitly defined",

        failures,

    )


    # ------------------------------------------------------------------------
    # Search PPU validation
    # ------------------------------------------------------------------------

    if search_ppu is not None:

        print(
            f"Search pixels/unit:     {search_ppu}"
        )

        check(

            abs(
                float(search_ppu)
                - EXPECTED_SEARCH_PPU
            ) <= TOL,

            f"Search sampling = "
            f"{EXPECTED_SEARCH_PPU} pixels/unit",

            failures,

        )


    # ------------------------------------------------------------------------
    # Reference PPU validation
    # ------------------------------------------------------------------------

    if reference_ppu is not None:

        print(
            f"Reference pixels/unit:  {reference_ppu}"
        )

        check(

            abs(
                float(reference_ppu)
                - EXPECTED_REFERENCE_PPU
            ) <= TOL,

            f"Reference sampling = "
            f"{EXPECTED_REFERENCE_PPU} pixels/unit",

            failures,

        )


    # ------------------------------------------------------------------------
    # Sampling ratio validation
    # ------------------------------------------------------------------------

    if (

        search_ppu is not None

        and reference_ppu is not None

    ):

        ratio = (

            float(reference_ppu)

            /

            float(search_ppu)

        )


        print(
            f"Sampling ratio:         {ratio:.6f}x"
        )


        check(

            abs(
                ratio
                - EXPECTED_SAMPLING_RATIO
            ) <= TOL,

            "Reference/Search sampling ratio = 10x",

            failures,

        )


    print()


    # ========================================================================
    # OBSERVATION GENERATION
    # ========================================================================

    print(
        "OBSERVATION SHAPES"
    )

    print(
        "-" * 76
    )


    try:

        obs_periodic = renderer.generate_observation(

            TX,

            TY,

            "periodic",

            SEED,

        )

    except Exception as exc:

        print(
            "[FAIL] Could not generate periodic observation"
        )

        print(
            exc
        )

        return 1


    search = obs_periodic["search"]

    reference = obs_periodic["reference"]

    template = obs_periodic["template"]


    print(
        f"Search:     {search.shape}"
    )

    print(
        f"Reference:  {reference.shape}"
    )

    print(
        f"Template:   {template.shape}"
    )


    # ------------------------------------------------------------------------
    # Search shape
    # ------------------------------------------------------------------------

    check(

        search.shape == EXPECTED_SEARCH_SHAPE,

        "Search image = 1000 x 1000",

        failures,

    )


    # ------------------------------------------------------------------------
    # Reference shape
    # ------------------------------------------------------------------------

    check(

        reference.shape == EXPECTED_REFERENCE_SHAPE,

        "Reference image = 1000 x 1000",

        failures,

    )


    # ------------------------------------------------------------------------
    # Search-equivalent template shape
    # ------------------------------------------------------------------------

    check(

        template.shape == EXPECTED_TEMPLATE_SHAPE,

        "Search-equivalent template = 100 x 100",

        failures,

    )


    print()


    # ========================================================================
    # NUMERICAL INTEGRITY
    # ========================================================================

    print(
        "NUMERICAL INTEGRITY"
    )

    print(
        "-" * 76
    )


    for name, arr in [

        ("Search", search),

        ("Reference", reference),

        ("Template", template),

    ]:

        # --------------------------------------------------------------------
        # NumPy array check
        # --------------------------------------------------------------------

        check(

            isinstance(
                arr,
                np.ndarray,
            ),

            f"{name} is a NumPy array",

            failures,

        )


        # --------------------------------------------------------------------
        # Finite-value check
        # --------------------------------------------------------------------

        check(

            np.isfinite(arr).all(),

            f"{name} contains only finite values",

            failures,

        )


    print()


    # ========================================================================
    # GROUND TRUTH
    # ========================================================================

    print(
        "GROUND-TRUTH"
    )

    print(
        "-" * 76
    )


    gt_x = obs_periodic.get(
        "gt_x"
    )

    gt_y = obs_periodic.get(
        "gt_y"
    )


    print(
        f"GT top-left: ({gt_x}, {gt_y})"
    )


    # ------------------------------------------------------------------------
    # GT existence
    # ------------------------------------------------------------------------

    check(

        gt_x is not None
        and gt_y is not None,

        "Ground-truth coordinates returned",

        failures,

    )


    if (

        gt_x is not None

        and gt_y is not None

    ):

        # --------------------------------------------------------------------
        # GT x integer
        # --------------------------------------------------------------------

        check(

            isinstance(
                gt_x,
                (
                    int,
                    np.integer,
                ),
            ),

            "GT x is integer",

            failures,

        )


        # --------------------------------------------------------------------
        # GT y integer
        # --------------------------------------------------------------------

        check(

            isinstance(
                gt_y,
                (
                    int,
                    np.integer,
                ),
            ),

            "GT y is integer",

            failures,

        )


        # --------------------------------------------------------------------
        # GT x bounds
        #
        # Search width = 1000
        # Template width = 100
        #
        # Therefore:
        #
        #   0 <= x <= 900
        # --------------------------------------------------------------------

        check(

            0 <= gt_x
            <= EXPECTED_SEARCH_SHAPE[1]
            - EXPECTED_TEMPLATE_SHAPE[1],

            "GT x keeps template inside search",

            failures,

        )


        # --------------------------------------------------------------------
        # GT y bounds
        #
        # Search height = 1000
        # Template height = 100
        #
        # Therefore:
        #
        #   0 <= y <= 900
        # --------------------------------------------------------------------

        check(

            0 <= gt_y
            <= EXPECTED_SEARCH_SHAPE[0]
            - EXPECTED_TEMPLATE_SHAPE[0],

            "GT y keeps template inside search",

            failures,

        )


    print()


    # ========================================================================
    # SCENE-TYPE CONSISTENCY
    # ========================================================================

    print(
        "SCENE-TYPE CONSISTENCY"
    )

    print(
        "-" * 76
    )


    for scene_type in [

        "periodic",

        "quasiperiodic",

    ]:

        try:

            obs = renderer.generate_observation(

                TX,

                TY,

                scene_type,

                SEED,

            )


            # ----------------------------------------------------------------
            # Search
            # ----------------------------------------------------------------

            check(

                obs["search"].shape
                == EXPECTED_SEARCH_SHAPE,

                f"{scene_type}: search shape correct",

                failures,

            )


            # ----------------------------------------------------------------
            # Reference
            # ----------------------------------------------------------------

            check(

                obs["reference"].shape
                == EXPECTED_REFERENCE_SHAPE,

                f"{scene_type}: reference shape correct",

                failures,

            )


            # ----------------------------------------------------------------
            # Template
            # ----------------------------------------------------------------

            check(

                obs["template"].shape
                == EXPECTED_TEMPLATE_SHAPE,

                f"{scene_type}: template shape correct",

                failures,

            )


        except Exception as exc:

            check(

                False,

                f"{scene_type}: observation generation succeeds",

                failures,

            )


            print(
                f"  Error: {exc}"
            )


    print()


    # ========================================================================
    # FINAL RESULT
    # ========================================================================

    print(
        "=" * 76
    )

    print(
        "STEP 33B RESULT"
    )

    print(
        "=" * 76
    )


    if failures:

        print()

        print(
            f"FAILED: {len(failures)} check(s)"
        )

        print()

        print(
            "Canonical acquisition model compliance failed."
        )

        print()

        print(
            "Failed checks:"
        )


        for failure in failures:

            print(
                f"  - {failure}"
            )


        print()

        return 1


    # ========================================================================
    # SUCCESS
    # ========================================================================

    print()

    print(
        "PASS"
    )

    print()

    print(
        "Canonical acquisition model is explicit and consistent."
    )

    print()

    print(
        "Search:       1000 x 1000"
    )

    print(
        "Reference:    1000 x 1000"
    )

    print(
        "Template:      100 x 100"
    )

    print(
        "Sampling ratio: 10x"
    )

    print()

    print(
        "Physical interpretation:"
    )

    print(
        "Search FOV:       200 x 200 physical units"
    )

    print(
        "Reference FOV:     20 x 20 physical units"
    )

    print(
        "Template FOV:      20 x 20 physical units"
    )

    print()

    print(
        "STEP 33B COMPLETE"
    )

    print()


    return 0


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":

    raise SystemExit(
        main()
    )