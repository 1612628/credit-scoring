import click

from src.data_mining.pipeline_manager import PipelineManager

pipeline_manager = PipelineManager()

@click.group()
def main():
    pass

@main.command()
@main.option(param_decls='-p', show_default='--pipe-line', help='predefined pipeline to be trained', required=True)
@main.option(param_decls='-d', show_default='--dev-mode', help='Development mode. If True then only small sample of data will be used', is_flag=True, required=False)
@main.option(param_decls='-t', show_default='--tag', help='Tagging', required=False)
def train(pipeline_name, dev_mode, tag):
    pipeline_manager.train(pipeline_name, dev_mode, tag)

@main.command()
@main.option(param_decls='-p', show_default='--pipe-line', help='predefined pipeline to be trained', required=True)
@main.option(param_decls='-d', show_default='--dev-mode', help='Development mode. If True then only small sample of data will be used', is_flag=True, required=False)
@main.option(param_decls='-t', show_default='--tag', help='Tagging', required=False)
def evaluate(pipeline_name, dev_mode, tag):
    pipeline_manager.evaluate(pipeline_name, dev_mode, tag)


if __name__ == '__main__':
    main()