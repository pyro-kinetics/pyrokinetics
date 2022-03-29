from pathlib import Path

template_dir = Path(__file__).parent / "templates"

gk_gs2_template = template_dir / "input.gs2"
gk_cgyro_template = template_dir / "input.cgyro"
gk_gene_template = template_dir / "input.gene"

eq_geqdsk_template = template_dir / "test.geqdsk"
eq_transp_template = template_dir / "transp_eq.cdf"

kinetics_scene_template = template_dir / "scene.cdf"
kinetics_jetto_template = template_dir / "jetto.cdf"
kinetics_transp_template = template_dir / "transp.cdf"
