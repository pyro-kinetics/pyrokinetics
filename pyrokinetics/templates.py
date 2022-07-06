from pathlib import Path

template_dir = Path(__file__).parent / "templates"
template_dir.resolve()

gk_gs2_template = template_dir / "input.gs2"
gk_cgyro_template = template_dir / "input.cgyro"
gk_gene_template = template_dir / "input.gene"
gk_tglf_template = template_dir / "input.tglf"
gk_templates = {
    "GS2": gk_gs2_template,
    "CGYRO": gk_cgyro_template,
    "GENE": gk_gene_template,
    "TGLF": gk_tglf_template,
}

eq_geqdsk_template = template_dir / "test.geqdsk"
eq_transp_template = template_dir / "transp_eq.cdf"
eq_templates = {
    "GEQDSK": eq_geqdsk_template,
    "TRANSP": eq_transp_template,
}

kinetics_scene_template = template_dir / "scene.cdf"
kinetics_jetto_template = template_dir / "jetto.cdf"
kinetics_transp_template = template_dir / "transp.cdf"
kinetics_templates = {
    "SCENE": kinetics_scene_template,
    "JETTO": kinetics_jetto_template,
    "TRANSP": kinetics_transp_template,
}
