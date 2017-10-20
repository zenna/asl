"""Architectures for Clevr Interfaces"""

def nmidseq(n, nhidden=1):
  "number of hidden layers for Equality function, n: size of type"
  return [2 * n for _ in range(nhidden)]

def funcs(arch,
         arch_opt,
         Unique,
         Relate,
         Count,
         Exist,
         Filter,
         FilterSize,
         FilterColor,
         FilterMaterial,
         FilterShape,
         Intersect,
         Union,
         GreaterThan,
         LessThan,
         EqualInteger,
         Equal,
         EqualMaterial,
         EqualSize,
         EqualShape,
         EqualColor,
         QueryShape,
         QuerySize,
         QueryMaterial,
         QueryColor,
         SameShape,
         SameSize,
         SameMaterial,
         SameColor):

  # eq_arch_opt = {'nmids': nmidseq()}
  # eq = Equal(arch=MLPNet, arch_opt=eq_arch_opt, sample=sample, sample_args=sample_args)
  #
  neu_clevr = {'unique': Unique(arch=arch, arch_opt=arch_opt, sample=sample, sample_args=sample_args),
               'relate': Relate(arch=arch, arch_opt=arch_opt, sample=sample, sample_args=sample_args),
               'count': Count(arch=arch, arch_opt=arch_opt, sample=sample, sample_args=sample_args),
               'exist': Exist(arch=arch, arch_opt=arch_opt, sample=sample, sample_args=sample_args),
               'intersect': Intersect(arch=arch, arch_opt=arch_opt, sample=sample, sample_args=sample_args),
               'union': Union(arch=arch, arch_opt=arch_opt, sample=sample, sample_args=sample_args),
               'greater_than': GreaterThan(arch=arch, arch_opt=arch_opt, sample=sample, sample_args=sample_args),
               'less_than': LessThan(arch=arch, arch_opt=arch_opt, sample=sample, sample_args=sample_args),
               'equal_integer': EqualInteger(arch=arch, arch_opt=arch_opt, sample=sample, sample_args=sample_args)}
  if share_funcs:
    fil = Filter(arch=arch, arch_opt=arch_opt, sample=sample, sample_args=sample_args)
    eq = Equal(arch=arch, arch_opt=arch_opt, sample=sample, sample_args=sample_args)
    query = Query(arch=arch, arch_opt=arch_opt, sample=sample, sample_args=sample_args)
    same = Same(arch=arch, arch_opt=arch_opt, sample=sample, sample_args=sample_args)
    neu_clevr.update({'filter_size': fil,
                      'filter_color': fil,
                      'filter_material': fil,
                      'filter_shape': fil,
                      'equal_material': eq,
                      'equal_size': eq,
                      'equal_shape': eq,
                      'equal_color': eq,
                      'query_shape': query,
                      'query_size': query,
                      'query_material': query,
                      'query_color': query,
                      'same_shape': same,
                      'same_size': same,
                      'same_material': same,
                      'same_color': same})
  else:
    neu_clevr.update({'filter_size': FilterSize(arch=arch, arch_opt=arch_opt, sample=sample, sample_args=sample_args),
                      'filter_color': FilterColor(arch=arch, arch_opt=arch_opt, sample=sample, sample_args=sample_args),
                      'filter_material': FilterMaterial(arch=arch, arch_opt=arch_opt, sample=sample, sample_args=sample_args),
                      'filter_shape': FilterShape(arch=arch, arch_opt=arch_opt, sample=sample, sample_args=sample_args),
                      'equal_material': EqualMaterial(arch=arch, arch_opt=arch_opt, sample=sample, sample_args=sample_args),
                      'equal_size': EqualSize(arch=arch, arch_opt=arch_opt, sample=sample, sample_args=sample_args),
                      'equal_shape': EqualShape(arch=arch, arch_opt=arch_opt, sample=sample, sample_args=sample_args),
                      'equal_color': EqualColor(arch=arch, arch_opt=arch_opt, sample=sample, sample_args=sample_args),
                      'query_shape': QueryShape(arch=arch, arch_opt=arch_opt, sample=sample, sample_args=sample_args),
                      'query_size': QuerySize(arch=arch, arch_opt=arch_opt, sample=sample, sample_args=sample_args),
                      'query_material': QueryMaterial(arch=arch, arch_opt=arch_opt, sample=sample, sample_args=sample_args),
                      'query_color': QueryColor(arch=arch, arch_opt=arch_opt, sample=sample, sample_args=sample_args),
                      'same_shape': SameShape(arch=arch, arch_opt=arch_opt, sample=sample, sample_args=sample_args),
                      'same_size': SameSize(arch=arch, arch_opt=arch_opt, sample=sample, sample_args=sample_args),
                      'same_material': SameMaterial(arch=arch, arch_opt=arch_opt, sample=sample, sample_args=sample_args),
                      'same_color': SameColor(arch=arch, arch_opt=arch_opt, sample=sample, sample_args=sample_args)})
    return neu_clevr
