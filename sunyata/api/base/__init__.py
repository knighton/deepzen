class API(object):
    def _api(self, api_obj):
        is_public = lambda s: not s.startswith('__')
        members_of = lambda x: list(filter(is_public, dir(x)))
        mine = members_of(self)
        theirs = members_of(api_obj)
        for method_name in filter(lambda s: s not in mine, theirs):
            setattr(self, method_name, getattr(api_obj, method_name))
