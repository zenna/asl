def observe_loss(criterion, obs, refobs, state=None):
  "MSE between observations from reference and training stack"
  total_loss = 0.0
  losses = [criterion(obs[i], refobs[i]) for i in range(len(obs))]
  total_loss = sum(losses) / len(losses)
  return total_loss
